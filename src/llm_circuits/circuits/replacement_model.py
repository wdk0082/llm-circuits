"""Replace MLP layers with transcoder reconstructions during forward passes.

Supports both per-layer transcoders (``TranscoderSet``) and cross-layer
transcoders (``CrossLayerTranscoder``).  The context manager
:func:`replace_mlps_with_transcoders` temporarily hooks each MLP submodule
so that its output is replaced by the corresponding transcoder reconstruction.

.. note::

    Transcoders cannot reliably reconstruct the first token position (the
    attention-sink / BOS position).  By default ``n_bos_tokens=1`` preserves
    the original MLP output at position 0.  **Callers should ensure the input
    starts with a BOS or padding token** so that position 0 acts as the
    attention sink and real content tokens occupy positions >= 1.

.. note::

    Some architectures read the transcoder input and write its output on
    *different* submodules (a "two-hook" layer).  Pass
    ``output_module_template="model.layers.{layer}.<output_submodule>"`` to
    :func:`replace_mlps_with_transcoders` and :func:`compare_models` to replace
    that separate output module.  Qwen3 is single-hook, so it is left ``None``.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from llm_circuits.instrumentation.hooks import ActivationRecorder
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


class _InputBuffer:
    """Stores captured transcoder inputs for two-hook architectures.

    When the transcoder's input source and output target live on different
    submodules, an input-capture hook stores the MLP input here, and the
    output-replacement hook consumes it.
    """

    def __init__(self) -> None:
        self._buf: dict[int, Tensor] = {}

    def store(self, layer: int, x: Tensor) -> None:
        self._buf[layer] = x

    def pop(self, layer: int) -> Tensor:
        return self._buf.pop(layer)

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

    top5_agreement: Tensor
    """Whether original top-1 is within replacement top-5 per position, shape ``(seq,)``."""

    original_logits: Tensor
    """Shape ``(seq, vocab)``."""

    replacement_logits: Tensor
    """Shape ``(seq, vocab)``."""

    reconstruction_errors: dict[int, Tensor]
    """Per-layer L2 norm of error over *d_model*, shape ``(seq,)``."""

    original_activations: dict[int, Tensor]
    """Per-layer original MLP output, shape ``(seq, d_model)``."""

    replacement_activations: dict[int, Tensor]
    """Per-layer transcoder reconstruction, shape ``(seq, d_model)``."""


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _is_transcoder_set(tc: Any) -> bool:
    """Return True for ``TranscoderSet`` (has ``.transcoders`` ModuleList)."""
    return hasattr(tc, "transcoders")


def _finalize_reconstruction(
    reconstruction: Tensor,
    original_out: Tensor,
    layer_idx: int,
    ctx: ReplacementContext,
    include_error: bool,
    n_bos_tokens: int,
) -> Tensor:
    """Preserve BOS positions and store reconstruction/error in *ctx*."""
    if n_bos_tokens > 0:
        reconstruction = torch.cat(
            [original_out[..., :n_bos_tokens, :], reconstruction[..., n_bos_tokens:, :]],
            dim=-2,
        )

    ctx.reconstructions[layer_idx] = reconstruction.detach()
    if include_error:
        ctx.errors[layer_idx] = original_out.detach() - reconstruction.detach()

    return reconstruction


def _replace_output(output: Any, replacement: Tensor) -> Any:
    """Swap the primary tensor in a module output, preserving tuple structure."""
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    return replacement


def _compute_clt_reconstruction(
    x: Tensor,
    layer_idx: int,
    clt: CrossLayerTranscoder,
    buf: _CrossLayerBuffer,
) -> Tensor:
    """Run CLT encode + decode for *layer_idx*, updating *buf* with future contributions."""
    features = clt.encode_layer(x, layer_idx)

    # W_dec shape: (d_transcoder, n_target_layers, d_model)
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

    return reconstruction


def _make_input_capture_hook(layer_idx: int, ibuf: _InputBuffer) -> Any:
    """Return a forward hook that captures ``inp[0]`` without modifying output."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        ibuf.store(layer_idx, inp[0])
        return output

    return hook


def _make_plt_hook(
    layer_idx: int,
    transcoder: TranscoderSet,
    ctx: ReplacementContext,
    include_error: bool,
    n_bos_tokens: int,
    ibuf: _InputBuffer | None = None,
) -> Any:
    """Return a forward hook that replaces output with per-layer transcoder reconstruction.

    When *ibuf* is provided (two-hook mode), the transcoder input is read from
    the buffer instead of from the hooked module's ``inp[0]``.
    """

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = ibuf.pop(layer_idx) if ibuf is not None else inp[0]
        original_out = output[0] if isinstance(output, tuple) else output

        reconstruction = transcoder.transcoders[layer_idx](x)
        reconstruction = _finalize_reconstruction(
            reconstruction, original_out, layer_idx, ctx, include_error, n_bos_tokens
        )
        return _replace_output(output, reconstruction)

    return hook


def _make_clt_hook(
    layer_idx: int,
    clt: CrossLayerTranscoder,
    buf: _CrossLayerBuffer,
    ctx: ReplacementContext,
    include_error: bool,
    n_bos_tokens: int,
    ibuf: _InputBuffer | None = None,
) -> Any:
    """Return a forward hook that replaces output with CLT reconstruction.

    When *ibuf* is provided (two-hook mode), the transcoder input is read from
    the buffer instead of from the hooked module's ``inp[0]``.
    """

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = ibuf.pop(layer_idx) if ibuf is not None else inp[0]
        original_out = output[0] if isinstance(output, tuple) else output

        reconstruction = _compute_clt_reconstruction(x, layer_idx, clt, buf)
        reconstruction = _finalize_reconstruction(
            reconstruction, original_out, layer_idx, ctx, include_error, n_bos_tokens
        )
        return _replace_output(output, reconstruction)

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
    n_bos_tokens: int = 1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
) -> Generator[ReplacementContext, None, None]:
    """Temporarily replace every MLP with its transcoder reconstruction.

    Args:
        model: The full language model (e.g. ``AutoModelForCausalLM``).
        transcoder: A ``TranscoderSet`` or ``CrossLayerTranscoder`` loaded
            via :func:`~llm_circuits.transcoders.circuit_tracer_loader.load_transcoder`.
        include_error: If ``True``, store ``original - reconstruction`` per layer
            in :attr:`ReplacementContext.errors`.
        n_bos_tokens: Number of leading token positions whose original MLP output
            is preserved (not replaced).  Defaults to ``1`` because transcoders
            cannot reconstruct the attention-sink position.  Set to ``0`` to
            replace all positions (not recommended).
        mlp_name_template: Python format string with a ``{layer}`` placeholder
            used to resolve the submodule whose ``inp[0]`` provides the
            transcoder input.
        output_module_template: If provided, a *separate* submodule whose output
            is replaced by the transcoder reconstruction.  Required for
            architectures where the transcoder output target differs from the
            input source (a "two-hook" layer).  Qwen3 is single-hook, so this is
            left ``None``.

    Yields:
        A :class:`ReplacementContext` whose ``reconstructions`` (and optionally
        ``errors``) dicts are populated during the forward pass.
    """
    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers
    two_hook = output_module_template is not None

    ctx = ReplacementContext()
    handles: list[torch.utils.hooks.RemovableHook] = []
    buf = _CrossLayerBuffer() if not is_set else None
    ibuf = _InputBuffer() if two_hook else None

    # Freeze all model and transcoder parameters
    frozen_params: list[tuple[nn.Parameter, bool]] = []
    for p in model.parameters():
        frozen_params.append((p, p.requires_grad))
        p.requires_grad_(False)
    for p in transcoder.parameters():
        frozen_params.append((p, p.requires_grad))
        p.requires_grad_(False)

    try:
        for i in range(n_layers):
            input_mod = model.get_submodule(mlp_name_template.format(layer=i))

            if two_hook:
                output_mod = model.get_submodule(output_module_template.format(layer=i))
                handles.append(input_mod.register_forward_hook(_make_input_capture_hook(i, ibuf)))
                if is_set:
                    handles.append(
                        output_mod.register_forward_hook(
                            _make_plt_hook(i, transcoder, ctx, include_error, n_bos_tokens, ibuf)
                        )
                    )
                else:
                    handles.append(
                        output_mod.register_forward_hook(
                            _make_clt_hook(
                                i, transcoder, buf, ctx, include_error, n_bos_tokens, ibuf
                            )
                        )
                    )
            else:
                if is_set:
                    handles.append(
                        input_mod.register_forward_hook(
                            _make_plt_hook(i, transcoder, ctx, include_error, n_bos_tokens)
                        )
                    )
                else:
                    handles.append(
                        input_mod.register_forward_hook(
                            _make_clt_hook(i, transcoder, buf, ctx, include_error, n_bos_tokens)
                        )
                    )

        yield ctx
    finally:
        for h in handles:
            h.remove()
        if buf is not None:
            buf.clear()
        if ibuf is not None:
            ibuf.clear()
        # Restore original requires_grad state
        for p, orig in frozen_params:
            p.requires_grad_(orig)


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


@torch.no_grad()
def compare_models(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    *,
    n_bos_tokens: int = 1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
) -> ComparisonResult:
    """Run the model with and without transcoder replacement and compare outputs.

    Args:
        model: The full language model.
        transcoder: Transcoder set or cross-layer transcoder.
        input_ids: Token ids, shape ``(batch, seq)`` or ``(seq,)``.
            Batch dimension is squeezed in the returned metrics.
        n_bos_tokens: Forwarded to :func:`replace_mlps_with_transcoders`.
        mlp_name_template: Forwarded to :func:`replace_mlps_with_transcoders`.
        output_module_template: Forwarded to :func:`replace_mlps_with_transcoders`.

    Returns:
        A :class:`ComparisonResult` with per-position metrics.
    """
    # --- Original forward pass (with hooks to capture activations) -------------
    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    # Record activations at the output module (= what the transcoder reconstructs).
    out_template = output_module_template or mlp_name_template
    record_names = [out_template.format(layer=i) for i in range(n_layers)]

    recorder = ActivationRecorder()
    with recorder.attach(model, record_names):
        original_logits = model(input_ids).logits

    # --- Replacement forward pass ---------------------------------------------
    with replace_mlps_with_transcoders(
        model,
        transcoder,
        include_error=True,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        output_module_template=output_module_template,
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

    # --- Top-5 agreement ------------------------------------------------------
    orig_top1 = original_logits.argmax(dim=-1)  # (seq,)
    repl_top5 = replacement_logits.topk(5, dim=-1).indices  # (seq, 5)
    top5 = (repl_top5 == orig_top1.unsqueeze(-1)).any(dim=-1)

    # --- Per-layer reconstruction error (L2 over d_model) ---------------------
    reconstruction_errors: dict[int, Tensor] = {}
    for layer_idx, err in rctx.errors.items():
        # err shape: (batch, seq, d_model) or (seq, d_model)
        if err.dim() == 3:
            err = err[0]
        reconstruction_errors[layer_idx] = err.norm(dim=-1)

    # --- Per-layer activations ------------------------------------------------
    original_acts: dict[int, Tensor] = {}
    for i, name in enumerate(record_names):
        if name in recorder.activations:
            act = recorder.activations[name]
            if act.dim() == 3:
                act = act[0]
            original_acts[i] = act

    replacement_acts: dict[int, Tensor] = {}
    for layer_idx, recon in rctx.reconstructions.items():
        if recon.dim() == 3:
            recon = recon[0]
        replacement_acts[layer_idx] = recon

    return ComparisonResult(
        kl_divergence=kl,
        cosine_similarity=cosine,
        top1_agreement=top1,
        top5_agreement=top5,
        original_logits=original_logits,
        replacement_logits=replacement_logits,
        reconstruction_errors=reconstruction_errors,
        original_activations=original_acts,
        replacement_activations=replacement_acts,
    )
