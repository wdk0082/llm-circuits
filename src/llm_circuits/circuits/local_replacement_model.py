"""Local replacement model conditioned on a specific input.

Like :func:`~llm_circuits.circuits.replacement_model.replace_mlps_with_transcoders`,
this replaces MLP layers with transcoder reconstructions.  In addition it:

* **Adds error nodes** — the reconstruction error (``original_mlp_out -
  transcoder_reconstruction``) is injected back into the residual stream so that
  the final logits are identical to the original model.  Individual error nodes
  can later be ablated for attribution studies.
* **Freezes attention weights and RMSNorm denominators** — this linearises the
  model in the residual stream (the only remaining nonlinearities are inside the
  transcoders).
* **Detaches reconstructions** — after the transcoder produces a
  reconstruction, the entire reconstruction tensor is detached so it
  becomes a constant (no gradient flows through encoder or decoder
  weights).  Feature post-activations are stored detached for reference
  only.  Error nodes are also detached constants, consistent with the
  other frozen components.

The main entry points are:

* :class:`LocalReplacementModel` — a context manager that installs frozen hooks
  and exposes a reusable :meth:`~LocalReplacementModel.forward` method.
* :func:`run_local_replacement` — convenience wrapper that captures constants
  and runs one local forward pass in a single call.

Scope: **Qwen3 only** (Gemma2 support deferred).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from llm_circuits.circuits.replacement_model import (
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
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LocalReplacementContext:
    """Results from :meth:`LocalReplacementModel.forward`."""

    logits: Tensor
    """Local replacement model logits, shape ``(seq, vocab)``."""

    original_logits: Tensor
    """Original model logits, shape ``(seq, vocab)``."""

    reconstructions: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer transcoder reconstruction (detached), shape ``(seq, d_model)``."""

    errors: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer error (detached constant), shape ``(seq, d_model)``."""

    features: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer feature post-activations (detached, for reference),
    shape ``(seq, d_transcoder)``."""


@dataclass
class CapturedConstants:
    """Quantities captured during the original (unmodified) forward pass."""

    original_logits: Tensor
    attn_weights: dict[int, Tensor] = field(default_factory=dict)
    rmsnorm_scales: dict[str, Tensor] = field(default_factory=dict)
    mlp_outputs: dict[int, Tensor] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 1 helpers — capture constants from the original forward
# ---------------------------------------------------------------------------


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


def capture_constants(
    model: nn.Module,
    input_ids: Tensor,
    *,
    n_layers: int,
    mlp_name_template: str,
    attn_name_template: str,
    layernorm_templates: list[str],
    final_norm_name: str,
) -> CapturedConstants:
    """Run the original model once and capture all needed constants."""
    caps = CapturedConstants(original_logits=torch.empty(0))
    handles: list[torch.utils.hooks.RemovableHook] = []

    # Temporarily force eager attention so weights are returned
    attn_impls: dict[int, str] = {}
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
            out = model(input_ids, output_attentions=True)
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
# CLT reconstruction with gradient-aware feature detach
# ---------------------------------------------------------------------------


def _compute_clt_reconstruction_local(
    x: Tensor,
    layer_idx: int,
    clt: CrossLayerTranscoder,
    buf: _CrossLayerBuffer,
    features_store: dict[int, Tensor],
) -> Tensor:
    """CLT encode + decode, storing features for reference.

    Like :func:`~llm_circuits.circuits.replacement_model._compute_clt_reconstruction`
    but stores a detached copy of the features in *features_store*.  The caller
    is responsible for detaching the returned reconstruction.
    """
    # Encode
    features = clt.encode_layer(x, layer_idx)

    # Store detached copy for reference (no gradient needed)
    features_store[layer_idx] = features.detach()

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


# ---------------------------------------------------------------------------
# Local transcoder hook factories
# ---------------------------------------------------------------------------


def _make_local_plt_hook(
    layer_idx: int,
    transcoder: TranscoderSet,
    reconstructions: dict[int, Tensor],
    errors: dict[int, Tensor],
    features_store: dict[int, Tensor],
    captured_mlp_outputs: dict[int, Tensor],
    include_error: bool,
    n_bos_tokens: int,
    ibuf: _InputBuffer | None = None,
) -> Any:
    """Per-layer transcoder hook with constant reconstruction and constant error nodes."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = ibuf.pop(layer_idx) if ibuf is not None else inp[0]
        original_out = output[0] if isinstance(output, tuple) else output

        single_tc = transcoder.transcoders[layer_idx]

        # 1. Encode → feature post-activations (no detach)
        features = single_tc.encode(x)
        features_store[layer_idx] = features.detach()  # store detached copy for reference

        # 2. Decode, then detach — reconstruction is a constant
        reconstruction = single_tc.decode(features, x).detach()

        # 3. Preserve BOS positions
        if n_bos_tokens > 0:
            reconstruction = torch.cat(
                [
                    original_out[..., :n_bos_tokens, :].detach(),
                    reconstruction[..., n_bos_tokens:, :],
                ],
                dim=-2,
            )

        reconstructions[layer_idx] = reconstruction

        # 4. Error node — detached constant
        error = (captured_mlp_outputs[layer_idx] - reconstruction).detach()
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
    features_store: dict[int, Tensor],
    captured_mlp_outputs: dict[int, Tensor],
    include_error: bool,
    n_bos_tokens: int,
    ibuf: _InputBuffer | None = None,
) -> Any:
    """Cross-layer transcoder hook with constant reconstruction and constant error nodes."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = ibuf.pop(layer_idx) if ibuf is not None else inp[0]
        original_out = output[0] if isinstance(output, tuple) else output

        reconstruction = _compute_clt_reconstruction_local(
            x, layer_idx, clt, buf, features_store
        ).detach()

        # Preserve BOS positions
        if n_bos_tokens > 0:
            reconstruction = torch.cat(
                [
                    original_out[..., :n_bos_tokens, :].detach(),
                    reconstruction[..., n_bos_tokens:, :],
                ],
                dim=-2,
            )

        reconstructions[layer_idx] = reconstruction

        # Error node — detached constant
        error = (captured_mlp_outputs[layer_idx] - reconstruction).detach()
        errors[layer_idx] = error

        result = reconstruction + error if include_error else reconstruction
        return _replace_output(output, result)

    return hook


# ---------------------------------------------------------------------------
# LocalReplacementModel — reusable context manager for local forward passes
# ---------------------------------------------------------------------------


class LocalReplacementModel:
    """Context manager that installs frozen hooks for local replacement forward passes.

    Usage::

        caps = capture_constants(model, input_ids, ...)
        with LocalReplacementModel(model, transcoder, caps, ...) as local_model:
            ctx = local_model.forward(input_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        transcoder: TranscoderSet | CrossLayerTranscoder,
        caps: CapturedConstants,
        *,
        include_error: bool = True,
        n_bos_tokens: int = 1,
        mlp_name_template: str = "model.layers.{layer}.mlp",
        output_module_template: str | None = None,
        attn_name_template: str = "model.layers.{layer}.self_attn",
        layernorm_templates: list[str] | None = None,
        final_norm_name: str = "model.norm",
    ) -> None:
        self._model = model
        self._transcoder = transcoder
        self._caps = caps
        self._include_error = include_error
        self._n_bos_tokens = n_bos_tokens
        self._mlp_name_template = mlp_name_template
        self._output_module_template = output_module_template
        self._attn_name_template = attn_name_template
        self._layernorm_templates = (
            layernorm_templates
            if layernorm_templates is not None
            else list(_QWEN3_LAYERNORM_TEMPLATES)
        )
        self._final_norm_name = final_norm_name

        self._is_set = _is_transcoder_set(transcoder)
        self._n_layers = len(transcoder) if self._is_set else transcoder.n_layers
        self._two_hook = output_module_template is not None

        # Mutable state populated by hooks (cleared between forward calls)
        self._reconstructions: dict[int, Tensor] = {}
        self._errors: dict[int, Tensor] = {}
        self._features_store: dict[int, Tensor] = {}
        self._buf: _CrossLayerBuffer | None = None
        self._ibuf: _InputBuffer | None = None

        # Teardown state
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._saved_forwards: dict[int, Any] = {}
        self._frozen_params: list[tuple[nn.Parameter, bool]] = []
        self._active = False

    def __enter__(self) -> LocalReplacementModel:
        self._buf = _CrossLayerBuffer() if not self._is_set else None
        self._ibuf = _InputBuffer() if self._two_hook else None

        # Freeze all model and transcoder parameters
        for p in self._model.parameters():
            self._frozen_params.append((p, p.requires_grad))
            p.requires_grad_(False)
        for p in self._transcoder.parameters():
            self._frozen_params.append((p, p.requires_grad))
            p.requires_grad_(False)

        # --- Frozen RMSNorm hooks ---
        for tmpl in self._layernorm_templates:
            for i in range(self._n_layers):
                name = tmpl.format(layer=i)
                mod = self._model.get_submodule(name)
                self._handles.append(
                    mod.register_forward_hook(
                        _make_frozen_rmsnorm_hook(name, self._caps.rmsnorm_scales)
                    )
                )
        final_mod = self._model.get_submodule(self._final_norm_name)
        self._handles.append(
            final_mod.register_forward_hook(
                _make_frozen_rmsnorm_hook(self._final_norm_name, self._caps.rmsnorm_scales)
            )
        )

        # --- Frozen attention ---
        for i in range(self._n_layers):
            attn_mod = self._model.get_submodule(self._attn_name_template.format(layer=i))
            self._saved_forwards[i] = attn_mod.forward
            attn_mod.forward = _make_frozen_attn_forward(attn_mod, self._caps.attn_weights[i])

        # --- Transcoder replacement hooks ---
        for i in range(self._n_layers):
            input_mod = self._model.get_submodule(self._mlp_name_template.format(layer=i))

            if self._two_hook:
                output_mod = self._model.get_submodule(
                    self._output_module_template.format(layer=i)
                )
                self._handles.append(
                    input_mod.register_forward_hook(_make_input_capture_hook(i, self._ibuf))
                )
                if self._is_set:
                    self._handles.append(
                        output_mod.register_forward_hook(
                            _make_local_plt_hook(
                                i,
                                self._transcoder,
                                self._reconstructions,
                                self._errors,
                                self._features_store,
                                self._caps.mlp_outputs,
                                self._include_error,
                                self._n_bos_tokens,
                                self._ibuf,
                            )
                        )
                    )
                else:
                    self._handles.append(
                        output_mod.register_forward_hook(
                            _make_local_clt_hook(
                                i,
                                self._transcoder,
                                self._buf,
                                self._reconstructions,
                                self._errors,
                                self._features_store,
                                self._caps.mlp_outputs,
                                self._include_error,
                                self._n_bos_tokens,
                                self._ibuf,
                            )
                        )
                    )
            else:
                if self._is_set:
                    self._handles.append(
                        input_mod.register_forward_hook(
                            _make_local_plt_hook(
                                i,
                                self._transcoder,
                                self._reconstructions,
                                self._errors,
                                self._features_store,
                                self._caps.mlp_outputs,
                                self._include_error,
                                self._n_bos_tokens,
                            )
                        )
                    )
                else:
                    self._handles.append(
                        input_mod.register_forward_hook(
                            _make_local_clt_hook(
                                i,
                                self._transcoder,
                                self._buf,
                                self._reconstructions,
                                self._errors,
                                self._features_store,
                                self._caps.mlp_outputs,
                                self._include_error,
                                self._n_bos_tokens,
                            )
                        )
                    )

        self._active = True
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

        for i, orig_fwd in self._saved_forwards.items():
            attn_mod = self._model.get_submodule(self._attn_name_template.format(layer=i))
            attn_mod.forward = orig_fwd
        self._saved_forwards.clear()

        if self._buf is not None:
            self._buf.clear()
        if self._ibuf is not None:
            self._ibuf.clear()

        for p, orig in self._frozen_params:
            p.requires_grad_(orig)
        self._frozen_params.clear()

        self._active = False

    def forward(self, input_ids: Tensor) -> LocalReplacementContext:
        """Run a single local replacement forward pass.

        May be called multiple times while inside the context manager.
        Each call clears per-forward state and returns a fresh
        :class:`LocalReplacementContext`.
        """
        if not self._active:
            raise RuntimeError(
                "forward() must be called inside the LocalReplacementModel context manager"
            )

        # Clear per-call state
        self._reconstructions.clear()
        self._errors.clear()
        self._features_store.clear()
        if self._buf is not None:
            self._buf.clear()
        if self._ibuf is not None:
            self._ibuf.clear()

        # Run the local forward pass
        local_logits = self._model(input_ids).logits

        # Squeeze batch dim
        original_logits = self._caps.original_logits
        if original_logits.dim() == 3:
            original_logits = original_logits[0]
        if local_logits.dim() == 3:
            local_logits = local_logits[0]

        reconstructions = dict(self._reconstructions)
        for k, v in reconstructions.items():
            if v.dim() == 3:
                reconstructions[k] = v[0]

        return LocalReplacementContext(
            logits=local_logits,
            original_logits=original_logits,
            reconstructions=reconstructions,
            errors=dict(self._errors),
            features=dict(self._features_store),
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def run_local_replacement(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    *,
    include_error: bool = True,
    n_bos_tokens: int = 1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
) -> LocalReplacementContext:
    """Run a local replacement model conditioned on *input_ids*.

    Convenience wrapper that captures constants and runs one local forward
    pass.  For multiple forward passes with the same frozen state, use
    :class:`LocalReplacementModel` directly.

    This performs two forward passes:

    1. **Capture pass** — run the original model (under ``no_grad``) to record
       attention weights, RMSNorm denominators, and MLP outputs.
    2. **Local pass** — run the model again with hooks that replace MLPs with
       transcoders, freeze attention and layernorms, and inject error nodes
       as detached constants.

    Args:
        model: The full language model (e.g. ``AutoModelForCausalLM``).
        transcoder: A ``TranscoderSet`` or ``CrossLayerTranscoder``.
        input_ids: Token ids, shape ``(batch, seq)`` or ``(seq,)``.
        include_error: If ``True``, add reconstruction error back so that
            logits match the original model.
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
        errors, and feature activations.
    """
    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    caps = capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )

    with LocalReplacementModel(
        model,
        transcoder,
        caps,
        include_error=include_error,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        output_module_template=output_module_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    ) as local_model:
        return local_model.forward(input_ids)
