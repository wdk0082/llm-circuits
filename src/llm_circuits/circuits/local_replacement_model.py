"""Local replacement model conditioned on a specific input.

Like :func:`~llm_circuits.circuits.replacement_model.replace_mlps_with_transcoders`,
this replaces MLP layers with transcoder reconstructions.  In addition it can:

* **Add error nodes** — the reconstruction error (``original_mlp_out -
  transcoder_reconstruction``) is injected back into the residual stream so that
  the final logits are identical to the original model.  Individual error nodes
  can later be ablated for attribution studies.
* **Freeze attention weights and RMSNorm denominators** — this linearises the
  model in the residual stream (the only remaining nonlinearities are inside the
  transcoders).

Scope: **Qwen3 only** (Gemma2 support deferred).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from llm_circuits.circuits.replacement_model import (
    _compute_clt_reconstruction,
    _CrossLayerBuffer,
    _InputBuffer,
    _is_transcoder_set,
    _make_input_capture_hook,
    _replace_output,
)
from llm_circuits.instrumentation.hooks import ActivationRecorder
from llm_circuits.logging import get_logger

if TYPE_CHECKING:
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
    from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default layernorm templates for Qwen3
# ---------------------------------------------------------------------------

_QWEN3_LAYERNORM_TEMPLATES: list[str] = [
    "model.layers.{layer}.input_layernorm",
    "model.layers.{layer}.post_attention_layernorm",
]

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class LocalReplacementContext:
    """Results from :func:`run_local_replacement`."""

    logits: Tensor
    """Local replacement model logits, shape ``(seq, vocab)``."""

    original_logits: Tensor
    """Original model logits, shape ``(seq, vocab)``."""

    reconstructions: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer transcoder reconstruction, shape ``(seq, d_model)``."""

    errors: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer error ``original_mlp_out - reconstruction``, shape ``(seq, d_model)``."""

    frozen_attn_weights: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer frozen attention patterns, shape ``(batch, n_heads, seq, seq)``."""

    frozen_rmsnorm_scales: dict[str, Tensor] = field(default_factory=dict)
    """Per-module frozen RMSNorm scale factors."""


# ---------------------------------------------------------------------------
# Phase 1 helpers — capture constants from the original forward
# ---------------------------------------------------------------------------


@dataclass
class _CapturedConstants:
    """Quantities captured during the original (unmodified) forward pass."""

    original_logits: Tensor
    attn_weights: dict[int, Tensor] = field(default_factory=dict)
    rmsnorm_scales: dict[str, Tensor] = field(default_factory=dict)
    mlp_outputs: dict[int, Tensor] = field(default_factory=dict)


def _make_rmsnorm_capture_hook(name: str, store: dict[str, Tensor]) -> Any:
    """Return a forward hook that captures the RMSNorm scaling factor."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], _output: Any) -> None:
        x = inp[0]
        variance = x.float().pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(variance + _mod.variance_epsilon)
        store[name] = scale.detach()

    return hook


def _make_attn_capture_hook(layer_idx: int, store: dict[int, Tensor]) -> Any:
    """Return a forward hook that captures attention weights from ``output[1]``."""

    def hook(_mod: nn.Module, _inp: tuple[Any, ...], output: Any) -> None:
        # Qwen3Attention returns (attn_output, attn_weights, past_kv)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            store[layer_idx] = output[1].detach()

    return hook


def _capture_constants(
    model: nn.Module,
    input_ids: Tensor,
    *,
    n_layers: int,
    mlp_name_template: str,
    attn_name_template: str,
    layernorm_templates: list[str],
    final_norm_name: str,
    freeze_attention: bool,
    freeze_layernorms: bool,
) -> _CapturedConstants:
    """Run the original model once and capture all needed constants."""
    caps = _CapturedConstants(original_logits=torch.empty(0))
    handles: list[torch.utils.hooks.RemovableHook] = []

    # Temporarily force eager attention so weights are returned
    attn_impls: dict[int, str] = {}
    if freeze_attention:
        for i in range(n_layers):
            attn_mod = model.get_submodule(attn_name_template.format(layer=i))
            old_impl = getattr(attn_mod, "config", None)
            if old_impl is not None:
                old_val = getattr(attn_mod.config, "_attn_implementation", None)
                if old_val is not None and old_val != "eager":
                    attn_impls[i] = old_val
                    attn_mod.config._attn_implementation = "eager"

            handles.append(
                attn_mod.register_forward_hook(_make_attn_capture_hook(i, caps.attn_weights))
            )

    # RMSNorm capture hooks
    if freeze_layernorms:
        for tmpl in layernorm_templates:
            for i in range(n_layers):
                name = tmpl.format(layer=i)
                mod = model.get_submodule(name)
                handles.append(
                    mod.register_forward_hook(_make_rmsnorm_capture_hook(name, caps.rmsnorm_scales))
                )
        # Final norm
        final_mod = model.get_submodule(final_norm_name)
        handles.append(
            final_mod.register_forward_hook(
                _make_rmsnorm_capture_hook(final_norm_name, caps.rmsnorm_scales)
            )
        )

    # MLP output capture via ActivationRecorder
    mlp_names = [mlp_name_template.format(layer=i) for i in range(n_layers)]
    recorder = ActivationRecorder()
    for name in mlp_names:
        submod = model.get_submodule(name)
        handle = submod.register_forward_hook(recorder._make_hook(name))
        handles.append(handle)

    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=freeze_attention)
        caps.original_logits = out.logits.detach()

        # Copy MLP outputs from recorder
        for i, name in enumerate(mlp_names):
            if name in recorder.activations:
                caps.mlp_outputs[i] = recorder.activations[name]
    finally:
        for h in handles:
            h.remove()
        # Restore attention implementations
        for i, old_val in attn_impls.items():
            attn_mod = model.get_submodule(attn_name_template.format(layer=i))
            attn_mod.config._attn_implementation = old_val

    return caps


# ---------------------------------------------------------------------------
# Phase 2 helpers — frozen hooks for the local forward
# ---------------------------------------------------------------------------


def _make_frozen_rmsnorm_hook(name: str, frozen_scales: dict[str, Tensor]) -> Any:
    """Return a forward hook that replaces RMSNorm with linear scaling."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], _output: Any) -> Tensor:
        x = inp[0]
        input_dtype = x.dtype
        scale = frozen_scales[name]
        return _mod.weight * (x.float() * scale).to(input_dtype)

    return hook


def _repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads for GQA. Equivalent to transformers.models.qwen3.modeling_qwen3."""
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


def _make_frozen_attn_forward(attn_mod: nn.Module, frozen_weights: Tensor) -> Any:
    """Return a replacement forward that uses frozen attention weights.

    Only computes V projection + weighted sum + O projection.
    Skips Q/K projection, RoPE, and softmax entirely.
    """
    # Cache module references
    v_proj = attn_mod.v_proj
    o_proj = attn_mod.o_proj
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    head_dim = attn_mod.head_dim

    def frozen_forward(
        hidden_states: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None, Any]:
        bsz, q_len, _ = hidden_states.shape

        # V projection only
        value_states = v_proj(hidden_states)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = _repeat_kv(value_states, n_rep)

        # Weighted sum with frozen attention weights
        attn_output = frozen_weights @ value_states  # (bsz, n_heads, seq, head_dim)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = o_proj(attn_output)

        return attn_output, frozen_weights

    return frozen_forward


# ---------------------------------------------------------------------------
# Local transcoder hook factories
# ---------------------------------------------------------------------------


def _make_local_plt_hook(
    layer_idx: int,
    transcoder: TranscoderSet,
    reconstructions: dict[int, Tensor],
    errors: dict[int, Tensor],
    captured_mlp_outputs: dict[int, Tensor],
    include_error: bool,
    n_bos_tokens: int,
    ibuf: _InputBuffer | None = None,
) -> Any:
    """Like ``_make_plt_hook`` but with error node support."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = ibuf.pop(layer_idx) if ibuf is not None else inp[0]
        original_out = output[0] if isinstance(output, tuple) else output

        reconstruction = transcoder.transcoders[layer_idx](x)

        # Preserve BOS positions
        if n_bos_tokens > 0:
            reconstruction = torch.cat(
                [original_out[..., :n_bos_tokens, :], reconstruction[..., n_bos_tokens:, :]],
                dim=-2,
            )

        reconstructions[layer_idx] = reconstruction.detach()
        error = captured_mlp_outputs[layer_idx] - reconstruction.detach()
        errors[layer_idx] = error

        result = reconstruction + error if include_error else reconstruction
        return _replace_output(output, result)

    return hook


def _make_local_clt_hook(
    layer_idx: int,
    clt: CrossLayerTranscoder,
    buf: _CrossLayerBuffer,
    reconstructions: dict[int, Tensor],
    errors: dict[int, Tensor],
    captured_mlp_outputs: dict[int, Tensor],
    include_error: bool,
    n_bos_tokens: int,
    ibuf: _InputBuffer | None = None,
) -> Any:
    """Like ``_make_clt_hook`` but with error node support."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = ibuf.pop(layer_idx) if ibuf is not None else inp[0]
        original_out = output[0] if isinstance(output, tuple) else output

        reconstruction = _compute_clt_reconstruction(x, layer_idx, clt, buf)

        # Preserve BOS positions
        if n_bos_tokens > 0:
            reconstruction = torch.cat(
                [original_out[..., :n_bos_tokens, :], reconstruction[..., n_bos_tokens:, :]],
                dim=-2,
            )

        reconstructions[layer_idx] = reconstruction.detach()
        error = captured_mlp_outputs[layer_idx] - reconstruction.detach()
        errors[layer_idx] = error

        result = reconstruction + error if include_error else reconstruction
        return _replace_output(output, result)

    return hook


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_local_replacement(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    *,
    include_error: bool = True,
    freeze_attention: bool = True,
    freeze_layernorms: bool = True,
    n_bos_tokens: int = 1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
) -> LocalReplacementContext:
    """Run a local replacement model conditioned on *input_ids*.

    This performs two forward passes:

    1. **Capture pass** — run the original model to record attention weights,
       RMSNorm denominators, and MLP outputs.
    2. **Local pass** — run the model again with hooks that replace MLPs with
       transcoders, optionally inject error nodes, and optionally freeze
       attention and layernorms.

    Args:
        model: The full language model (e.g. ``AutoModelForCausalLM``).
        transcoder: A ``TranscoderSet`` or ``CrossLayerTranscoder``.
        input_ids: Token ids, shape ``(batch, seq)`` or ``(seq,)``.
        include_error: If ``True``, add reconstruction error back so that
            logits match the original model.
        freeze_attention: If ``True``, freeze attention weight matrices
            (softmax outputs) from the capture pass.
        freeze_layernorms: If ``True``, freeze RMSNorm denominators from
            the capture pass.
        n_bos_tokens: Number of leading positions to preserve original MLP
            output (transcoders cannot reconstruct the attention-sink position).
        mlp_name_template: Format string for MLP submodule names.
        output_module_template: Separate output module (for Gemma2-style
            architectures).
        attn_name_template: Format string for attention submodule names.
        layernorm_templates: Format strings for RMSNorm modules to freeze.
            Defaults to Qwen3 templates.
        final_norm_name: Name of the final layer norm module.

    Returns:
        A :class:`LocalReplacementContext` with logits, reconstructions,
        errors, and frozen constants.
    """
    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers
    two_hook = output_module_template is not None

    # ------------------------------------------------------------------
    # Phase 1: Capture constants from the original forward
    # ------------------------------------------------------------------
    caps = _capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
        freeze_attention=freeze_attention,
        freeze_layernorms=freeze_layernorms,
    )

    # ------------------------------------------------------------------
    # Phase 2: Local forward with hooks
    # ------------------------------------------------------------------
    handles: list[torch.utils.hooks.RemovableHook] = []
    saved_forwards: dict[int, Any] = {}
    reconstructions: dict[int, Tensor] = {}
    errors: dict[int, Tensor] = {}
    buf = _CrossLayerBuffer() if not is_set else None
    ibuf = _InputBuffer() if two_hook else None

    try:
        # --- Frozen RMSNorm hooks ---
        if freeze_layernorms:
            for tmpl in layernorm_templates:
                for i in range(n_layers):
                    name = tmpl.format(layer=i)
                    mod = model.get_submodule(name)
                    handles.append(
                        mod.register_forward_hook(
                            _make_frozen_rmsnorm_hook(name, caps.rmsnorm_scales)
                        )
                    )
            final_mod = model.get_submodule(final_norm_name)
            handles.append(
                final_mod.register_forward_hook(
                    _make_frozen_rmsnorm_hook(final_norm_name, caps.rmsnorm_scales)
                )
            )

        # --- Frozen attention ---
        if freeze_attention:
            for i in range(n_layers):
                attn_mod = model.get_submodule(attn_name_template.format(layer=i))
                saved_forwards[i] = attn_mod.forward
                attn_mod.forward = _make_frozen_attn_forward(attn_mod, caps.attn_weights[i])

        # --- Transcoder replacement hooks ---
        for i in range(n_layers):
            input_mod = model.get_submodule(mlp_name_template.format(layer=i))

            if two_hook:
                output_mod = model.get_submodule(output_module_template.format(layer=i))
                handles.append(input_mod.register_forward_hook(_make_input_capture_hook(i, ibuf)))
                if is_set:
                    handles.append(
                        output_mod.register_forward_hook(
                            _make_local_plt_hook(
                                i,
                                transcoder,
                                reconstructions,
                                errors,
                                caps.mlp_outputs,
                                include_error,
                                n_bos_tokens,
                                ibuf,
                            )
                        )
                    )
                else:
                    handles.append(
                        output_mod.register_forward_hook(
                            _make_local_clt_hook(
                                i,
                                transcoder,
                                buf,
                                reconstructions,
                                errors,
                                caps.mlp_outputs,
                                include_error,
                                n_bos_tokens,
                                ibuf,
                            )
                        )
                    )
            else:
                if is_set:
                    handles.append(
                        input_mod.register_forward_hook(
                            _make_local_plt_hook(
                                i,
                                transcoder,
                                reconstructions,
                                errors,
                                caps.mlp_outputs,
                                include_error,
                                n_bos_tokens,
                            )
                        )
                    )
                else:
                    handles.append(
                        input_mod.register_forward_hook(
                            _make_local_clt_hook(
                                i,
                                transcoder,
                                buf,
                                reconstructions,
                                errors,
                                caps.mlp_outputs,
                                include_error,
                                n_bos_tokens,
                            )
                        )
                    )

        # Run the local forward pass
        local_logits = model(input_ids).logits.detach()

    finally:
        for h in handles:
            h.remove()
        # Restore original attention forwards
        for i, orig_fwd in saved_forwards.items():
            attn_mod = model.get_submodule(attn_name_template.format(layer=i))
            attn_mod.forward = orig_fwd
        if buf is not None:
            buf.clear()
        if ibuf is not None:
            ibuf.clear()

    # Squeeze batch dim
    original_logits = caps.original_logits
    if original_logits.dim() == 3:
        original_logits = original_logits[0]
    if local_logits.dim() == 3:
        local_logits = local_logits[0]
    for d in (reconstructions, errors):
        for k, v in d.items():
            if v.dim() == 3:
                d[k] = v[0]

    return LocalReplacementContext(
        logits=local_logits,
        original_logits=original_logits,
        reconstructions=reconstructions,
        errors=errors,
        frozen_attn_weights=caps.attn_weights,
        frozen_rmsnorm_scales=caps.rmsnorm_scales,
    )
