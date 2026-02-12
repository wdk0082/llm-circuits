"""Replace MLP layers with transcoder reconstructions during forward passes.

Supports both per-layer transcoders (``TranscoderSet``) and cross-layer
transcoders (``CrossLayerTranscoder``).  The context manager
:func:`replace_mlps_with_transcoders` temporarily hooks each MLP submodule
so that its output is replaced by the corresponding transcoder reconstruction.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from llm_circuits.logging import get_logger

if TYPE_CHECKING:
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
    from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _CrossLayerBuffer:
    """Accumulates future-layer decoder contributions for cross-layer transcoders.

    During a sequential forward pass, features encoded at layer *j* contribute
    to layers ``j, j+1, ..., n_layers-1``.  Contributions destined for future
    layers are stored here and consumed when the target layer's hook fires.
    """

    def __init__(self) -> None:
        self._buf: dict[int, Tensor] = {}

    def add(self, target_layer: int, contribution: Tensor) -> None:
        if target_layer in self._buf:
            self._buf[target_layer] = self._buf[target_layer] + contribution
        else:
            self._buf[target_layer] = contribution

    def pop(self, layer: int) -> Tensor | None:
        return self._buf.pop(layer, None)

    def clear(self) -> None:
        self._buf.clear()


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReplacementContext:
    """State populated during a :func:`replace_mlps_with_transcoders` block."""

    logits: Tensor | None = None
    """Set by the caller after running ``model(**inputs)``."""

    reconstructions: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer transcoder reconstruction (detached)."""

    errors: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer ``mlp_out - reconstruction`` (only when *include_error* is True)."""


@dataclass
class ComparisonResult:
    """Metrics comparing original model outputs to replacement-model outputs."""

    kl_divergence: Tensor
    """KL(P_orig || P_repl) per sequence position, shape ``(seq,)``."""

    cosine_similarity: Tensor
    """Cosine similarity of logit vectors per position, shape ``(seq,)``."""

    top1_agreement: Tensor
    """Whether top-1 predictions match per position, shape ``(seq,)``."""

    original_logits: Tensor
    """Shape ``(seq, vocab)``."""

    replacement_logits: Tensor
    """Shape ``(seq, vocab)``."""

    reconstruction_errors: dict[int, Tensor]
    """Per-layer L2 norm of error over *d_model*, shape ``(seq,)``."""


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _is_transcoder_set(tc: Any) -> bool:
    """Return True for ``TranscoderSet`` (has ``.transcoders`` ModuleList)."""
    return hasattr(tc, "transcoders")


def _make_per_layer_hook(
    layer_idx: int,
    transcoder: TranscoderSet,
    ctx: ReplacementContext,
    include_error: bool,
) -> Any:
    """Return a forward hook that replaces MLP output with per-layer transcoder."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = inp[0]
        reconstruction = transcoder.transcoders[layer_idx](x)

        # Handle tuple outputs (some models return (hidden, cache, ...))
        original_out = output[0] if isinstance(output, tuple) else output

        ctx.reconstructions[layer_idx] = reconstruction.detach()
        if include_error:
            ctx.errors[layer_idx] = (original_out.detach() - reconstruction.detach())

        if isinstance(output, tuple):
            return (reconstruction, *output[1:])
        return reconstruction

    return hook


def _make_cross_layer_hook(
    layer_idx: int,
    clt: CrossLayerTranscoder,
    buf: _CrossLayerBuffer,
    ctx: ReplacementContext,
    include_error: bool,
) -> Any:
    """Return a forward hook that replaces MLP output with CLT reconstruction."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = inp[0]

        # Encode features at this layer
        features = clt.encode_layer(x, layer_idx)

        # Decode: W_dec shape is (d_transcoder, n_target_layers, d_model)
        W_dec = clt._get_decoder_vectors(layer_idx)

        # Self-contribution (offset 0 = this layer)
        self_contrib = torch.einsum("...f,fd->...d", features, W_dec[:, 0, :])

        # Future-layer contributions
        for offset in range(1, W_dec.shape[1]):
            future_contrib = torch.einsum("...f,fd->...d", features, W_dec[:, offset, :])
            buf.add(layer_idx + offset, future_contrib)

        # Assemble reconstruction
        reconstruction = clt.b_dec[layer_idx] + self_contrib

        buffered = buf.pop(layer_idx)
        if buffered is not None:
            reconstruction = reconstruction + buffered

        if clt.skip_connection:
            reconstruction = reconstruction + clt.compute_skip(layer_idx, x)

        # Handle tuple outputs
        original_out = output[0] if isinstance(output, tuple) else output

        ctx.reconstructions[layer_idx] = reconstruction.detach()
        if include_error:
            ctx.errors[layer_idx] = (original_out.detach() - reconstruction.detach())

        if isinstance(output, tuple):
            return (reconstruction, *output[1:])
        return reconstruction

    return hook


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def replace_mlps_with_transcoders(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    *,
    include_error: bool = False,
    mlp_name_template: str = "model.layers.{layer}.mlp",
) -> Generator[ReplacementContext, None, None]:
    """Temporarily replace every MLP with its transcoder reconstruction.

    Args:
        model: The full language model (e.g. ``AutoModelForCausalLM``).
        transcoder: A ``TranscoderSet`` or ``CrossLayerTranscoder`` loaded
            via :func:`~llm_circuits.transcoders.circuit_tracer_loader.load_transcoder`.
        include_error: If ``True``, store ``mlp_out - reconstruction`` per layer
            in :attr:`ReplacementContext.errors`.
        mlp_name_template: Python format string with a ``{layer}`` placeholder
            used to resolve each MLP submodule.

    Yields:
        A :class:`ReplacementContext` whose ``reconstructions`` (and optionally
        ``errors``) dicts are populated during the forward pass.
    """
    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    ctx = ReplacementContext()
    handles: list[torch.utils.hooks.RemovableHook] = []
    buf = _CrossLayerBuffer() if not is_set else None

    try:
        for i in range(n_layers):
            mlp_name = mlp_name_template.format(layer=i)
            submodule = model.get_submodule(mlp_name)

            if is_set:
                hook_fn = _make_per_layer_hook(i, transcoder, ctx, include_error)
            else:
                hook_fn = _make_cross_layer_hook(i, transcoder, buf, ctx, include_error)

            handles.append(submodule.register_forward_hook(hook_fn))

        yield ctx
    finally:
        for h in handles:
            h.remove()
        if buf is not None:
            buf.clear()


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


@torch.no_grad()
def compare_models(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    *,
    mlp_name_template: str = "model.layers.{layer}.mlp",
) -> ComparisonResult:
    """Run the model with and without transcoder replacement and compare outputs.

    Args:
        model: The full language model.
        transcoder: Transcoder set or cross-layer transcoder.
        input_ids: Token ids, shape ``(batch, seq)`` or ``(seq,)``.
            Batch dimension is squeezed in the returned metrics.
        mlp_name_template: Forwarded to :func:`replace_mlps_with_transcoders`.

    Returns:
        A :class:`ComparisonResult` with per-position metrics.
    """
    # --- Original forward pass ------------------------------------------------
    original_logits = model(input_ids).logits

    # --- Replacement forward pass ---------------------------------------------
    with replace_mlps_with_transcoders(
        model, transcoder, include_error=True, mlp_name_template=mlp_name_template
    ) as rctx:
        replacement_logits = model(input_ids).logits

    # Squeeze batch dim if present (single example)
    if original_logits.dim() == 3:
        original_logits = original_logits[0]
        replacement_logits = replacement_logits[0]

    # --- KL divergence: KL(P_orig || P_repl) ---------------------------------
    log_p = F.log_softmax(original_logits, dim=-1)
    log_q = F.log_softmax(replacement_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)

    # --- Cosine similarity ----------------------------------------------------
    cosine = F.cosine_similarity(original_logits, replacement_logits, dim=-1)

    # --- Top-1 agreement ------------------------------------------------------
    top1 = original_logits.argmax(dim=-1) == replacement_logits.argmax(dim=-1)

    # --- Per-layer reconstruction error (L2 over d_model) ---------------------
    reconstruction_errors: dict[int, Tensor] = {}
    for layer_idx, err in rctx.errors.items():
        # err shape: (batch, seq, d_model) or (seq, d_model)
        if err.dim() == 3:
            err = err[0]
        reconstruction_errors[layer_idx] = err.norm(dim=-1)

    return ComparisonResult(
        kl_divergence=kl,
        cosine_similarity=cosine,
        top1_agreement=top1,
        original_logits=original_logits,
        replacement_logits=replacement_logits,
        reconstruction_errors=reconstruction_errors,
    )
