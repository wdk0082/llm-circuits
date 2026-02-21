"""Attribution graph built on the fully linearized local replacement model.

Constructs a graph of **nodes** (features, errors, input embeddings, output
logits) connected by **edges** whose weights are ``activation * virtual_weight``.

Virtual weights are computed efficiently via autograd backward passes through
the linearized model, avoiding explicit Jacobian materialisation.

Scope: **Qwen3 only** (matching :mod:`local_replacement_model`).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from llm_circuits.circuits.local_replacement_model import (
    _QWEN3_LAYERNORM_TEMPLATES,
    _capture_constants,
    _CapturedConstants,
    _make_frozen_attn_forward,
    _make_frozen_rmsnorm_hook,
)
from llm_circuits.circuits.replacement_model import (
    _CrossLayerBuffer,
    _is_transcoder_set,
    _replace_output,
)
from llm_circuits.logging import get_logger

if TYPE_CHECKING:
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
    from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node / edge data structures
# ---------------------------------------------------------------------------


class NodeType(enum.Enum):
    FEATURE = "feature"
    ERROR = "error"
    EMBEDDING = "embedding"
    LOGIT = "logit"


@dataclass(frozen=True)
class NodeId:
    """Unique identifier for a node in the attribution graph."""

    node_type: NodeType
    layer: int
    """Layer index. ``-1`` for embedding, ``n_layers`` for logit."""
    position: int
    feature_idx: int
    """Feature index within the transcoder. ``-1`` for non-feature nodes."""


@dataclass
class Node:
    """A node in the attribution graph."""

    id: NodeId
    activation: float


@dataclass
class AttributionGraph:
    """Complete attribution graph for a single prompt."""

    nodes: dict[NodeId, Node] = field(default_factory=dict)
    edges: list[tuple[NodeId, NodeId, float]] = field(default_factory=list)
    """Each edge is ``(source_id, target_id, weight)``."""
    tokens: list[str] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    model_name: str = ""
    transcoder_repo: str = ""

    # -- serialisation -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the graph via :func:`torch.save`."""

        def _nid(n: NodeId) -> tuple:
            return (n.node_type.value, n.layer, n.position, n.feature_idx)

        data = {
            "nodes": {_nid(n.id): n.activation for n in self.nodes.values()},
            "edges": [(_nid(s), _nid(t), w) for s, t, w in self.edges],
            "tokens": self.tokens,
            "input_ids": self.input_ids,
            "model_name": self.model_name,
            "transcoder_repo": self.transcoder_repo,
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str | Path) -> AttributionGraph:
        """Load a previously saved graph."""
        data = torch.load(path, weights_only=False)

        def _from_tuple(t: tuple) -> NodeId:
            return NodeId(NodeType(t[0]), t[1], t[2], t[3])

        nodes = {
            _from_tuple(k): Node(id=_from_tuple(k), activation=v) for k, v in data["nodes"].items()
        }
        edges = [(_from_tuple(s), _from_tuple(t), w) for s, t, w in data["edges"]]
        return cls(
            nodes=nodes,
            edges=edges,
            tokens=data["tokens"],
            input_ids=data["input_ids"],
            model_name=data.get("model_name", ""),
            transcoder_repo=data.get("transcoder_repo", ""),
        )


# ---------------------------------------------------------------------------
# Internal forward-pass result
# ---------------------------------------------------------------------------


@dataclass
class _ForwardResult:
    """Tensors retained from the gradient-enabled linearized forward pass."""

    logits: Tensor
    """``(seq, vocab)`` with autograd history."""

    feature_acts: dict[int, Tensor] = field(default_factory=dict)
    """``layer -> (seq, d_transcoder)`` leaf tensors with ``requires_grad``."""

    active_masks: dict[int, Tensor] = field(default_factory=dict)
    """``layer -> (seq, d_transcoder)`` boolean masks."""

    error_scales: dict[int, Tensor] = field(default_factory=dict)
    """``layer -> (seq, 1)`` leaf scalars multiplied with detached error vectors."""

    embed_scale: Tensor | None = None
    """``(seq, 1)`` leaf scalar multiplied with detached embeddings."""

    errors_detached: dict[int, Tensor] = field(default_factory=dict)
    """``layer -> (seq, d_model)`` detached error vectors."""

    embeddings_detached: Tensor | None = None
    """``(seq, d_model)`` detached token embeddings."""


# ---------------------------------------------------------------------------
# Hook factories for the gradient-enabled forward pass
# ---------------------------------------------------------------------------


def _make_feature_leaf_plt_hook(
    layer_idx: int,
    transcoder: TranscoderSet,
    fwd: _ForwardResult,
    captured_mlp_outputs: dict[int, Tensor],
    n_bos_tokens: int,
) -> Any:
    """MLP replacement hook that makes feature activations differentiable leaves.

    For :class:`TranscoderSet` (per-layer transcoders).
    """

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = inp[0]
        original_out = output[0] if isinstance(output, tuple) else output

        tc = transcoder.transcoders[layer_idx]

        # Encode
        pre_acts = F.linear(x.to(tc.W_enc.dtype), tc.W_enc, tc.b_enc)
        acts_raw = tc.activation_function(pre_acts)

        # Create leaf tensor — cuts the autograd graph at this feature boundary
        acts_leaf = acts_raw.detach().clone().requires_grad_(True)
        fwd.feature_acts[layer_idx] = acts_leaf
        fwd.active_masks[layer_idx] = acts_raw.detach() > 0

        # Decode from the leaf
        reconstruction = acts_leaf @ tc.W_dec + tc.b_dec
        if tc.W_skip is not None:
            reconstruction = reconstruction + x.detach() @ tc.W_skip.T

        # Preserve BOS positions
        if n_bos_tokens > 0:
            reconstruction = torch.cat(
                [
                    original_out[..., :n_bos_tokens, :].detach(),
                    reconstruction[..., n_bos_tokens:, :],
                ],
                dim=-2,
            )

        # Error as a scaled leaf
        recon_for_error = acts_raw.detach() @ tc.W_dec + tc.b_dec
        if tc.W_skip is not None:
            recon_for_error = recon_for_error + x.detach() @ tc.W_skip.T
        if n_bos_tokens > 0:
            recon_for_error = torch.cat(
                [
                    original_out[..., :n_bos_tokens, :].detach(),
                    recon_for_error[..., n_bos_tokens:, :],
                ],
                dim=-2,
            )

        error_vec = captured_mlp_outputs[layer_idx] - recon_for_error.detach()
        fwd.errors_detached[layer_idx] = error_vec.detach()

        scale = torch.ones(
            error_vec.shape[-2],
            1,
            device=error_vec.device,
            dtype=error_vec.dtype,
            requires_grad=True,
        )
        fwd.error_scales[layer_idx] = scale

        result = reconstruction + error_vec.detach() * scale
        return _replace_output(output, result)

    return hook


def _make_feature_leaf_clt_hook(
    layer_idx: int,
    clt: CrossLayerTranscoder,
    buf: _CrossLayerBuffer,
    fwd: _ForwardResult,
    captured_mlp_outputs: dict[int, Tensor],
    n_bos_tokens: int,
) -> Any:
    """MLP replacement hook with leaf activations for :class:`CrossLayerTranscoder`."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
        x = inp[0]
        original_out = output[0] if isinstance(output, tuple) else output

        # Encode
        pre_acts = clt.encode_layer(
            x.to(clt.W_enc.dtype), layer_idx, apply_activation_function=False
        )
        acts_raw = clt.apply_activation_function(layer_idx, pre_acts)

        # Leaf activations
        acts_leaf = acts_raw.detach().clone().requires_grad_(True)
        fwd.feature_acts[layer_idx] = acts_leaf
        fwd.active_masks[layer_idx] = acts_raw.detach() > 0

        # Decode from leaf — mirrors _compute_clt_reconstruction but with acts_leaf
        W_dec = clt._get_decoder_vectors(layer_idx)
        self_contrib = torch.einsum("...f,fd->...d", acts_leaf, W_dec[:, 0, :])

        for offset in range(1, W_dec.shape[1]):
            future_contrib = torch.einsum("...f,fd->...d", acts_leaf, W_dec[:, offset, :])
            buf.add(layer_idx + offset, future_contrib)

        reconstruction = clt.b_dec[layer_idx] + self_contrib

        buffered = buf.pop(layer_idx)
        if buffered is not None:
            reconstruction = reconstruction + buffered

        if clt.skip_connection:
            reconstruction = reconstruction + clt.compute_skip(layer_idx, x.detach())

        # BOS preservation
        if n_bos_tokens > 0:
            reconstruction = torch.cat(
                [
                    original_out[..., :n_bos_tokens, :].detach(),
                    reconstruction[..., n_bos_tokens:, :],
                ],
                dim=-2,
            )

        # Error
        from llm_circuits.circuits.replacement_model import _compute_clt_reconstruction

        # Need a separate buffer for the detached reconstruction
        recon_detached_buf = _CrossLayerBuffer()
        recon_detached = _compute_clt_reconstruction(x.detach(), layer_idx, clt, recon_detached_buf)
        if n_bos_tokens > 0:
            recon_detached = torch.cat(
                [
                    original_out[..., :n_bos_tokens, :].detach(),
                    recon_detached[..., n_bos_tokens:, :],
                ],
                dim=-2,
            )

        error_vec = captured_mlp_outputs[layer_idx] - recon_detached.detach()
        fwd.errors_detached[layer_idx] = error_vec.detach()

        scale = torch.ones(
            error_vec.shape[-2],
            1,
            device=error_vec.device,
            dtype=error_vec.dtype,
            requires_grad=True,
        )
        fwd.error_scales[layer_idx] = scale

        result = reconstruction + error_vec.detach() * scale
        return _replace_output(output, result)

    return hook


def _make_embed_scale_hook(fwd: _ForwardResult) -> Any:
    """Hook on the embedding module to wrap output with a differentiable scale."""

    def hook(_mod: nn.Module, _inp: tuple[Any, ...], output: Any) -> Any:
        embed = output[0] if isinstance(output, tuple) else output
        fwd.embeddings_detached = embed.detach()

        scale = torch.ones(
            embed.shape[-2],
            1,
            device=embed.device,
            dtype=embed.dtype,
            requires_grad=True,
        )
        fwd.embed_scale = scale

        scaled = embed.detach() * scale
        if isinstance(output, tuple):
            return (scaled, *output[1:])
        return scaled

    return hook


# ---------------------------------------------------------------------------
# Gradient-enabled linearized forward pass
# ---------------------------------------------------------------------------


def _linearized_forward_with_features(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    caps: _CapturedConstants,
    *,
    n_bos_tokens: int = 1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
    embed_module_name: str = "model.embed_tokens",
) -> _ForwardResult:
    """Run the linearized model with feature activations as autograd leaves."""

    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    fwd = _ForwardResult(logits=torch.empty(0))
    handles: list[torch.utils.hooks.RemovableHook] = []
    saved_forwards: dict[int, Any] = {}
    buf = _CrossLayerBuffer() if not is_set else None

    try:
        # --- Embedding scale hook ---
        embed_mod = model.get_submodule(embed_module_name)
        handles.append(embed_mod.register_forward_hook(_make_embed_scale_hook(fwd)))

        # --- Frozen RMSNorm hooks ---
        for tmpl in layernorm_templates:
            for i in range(n_layers):
                name = tmpl.format(layer=i)
                mod = model.get_submodule(name)
                handles.append(
                    mod.register_forward_hook(_make_frozen_rmsnorm_hook(name, caps.rmsnorm_scales))
                )
        final_mod = model.get_submodule(final_norm_name)
        handles.append(
            final_mod.register_forward_hook(
                _make_frozen_rmsnorm_hook(final_norm_name, caps.rmsnorm_scales)
            )
        )

        # --- Frozen attention ---
        for i in range(n_layers):
            attn_mod = model.get_submodule(attn_name_template.format(layer=i))
            saved_forwards[i] = attn_mod.forward
            attn_mod.forward = _make_frozen_attn_forward(attn_mod, caps.attn_weights[i])

        # --- Transcoder hooks with leaf activations ---
        for i in range(n_layers):
            mlp_mod = model.get_submodule(mlp_name_template.format(layer=i))
            if is_set:
                handles.append(
                    mlp_mod.register_forward_hook(
                        _make_feature_leaf_plt_hook(
                            i, transcoder, fwd, caps.mlp_outputs, n_bos_tokens
                        )
                    )
                )
            else:
                handles.append(
                    mlp_mod.register_forward_hook(
                        _make_feature_leaf_clt_hook(
                            i, transcoder, buf, fwd, caps.mlp_outputs, n_bos_tokens
                        )
                    )
                )

        # Forward pass WITH gradients
        fwd.logits = model(input_ids).logits

    finally:
        for h in handles:
            h.remove()
        for i, orig_fwd in saved_forwards.items():
            attn_mod = model.get_submodule(attn_name_template.format(layer=i))
            attn_mod.forward = orig_fwd
        if buf is not None:
            buf.clear()

    # Squeeze batch dimension
    if fwd.logits.dim() == 3:
        fwd.logits = fwd.logits[0]
    for layer_dict in (fwd.feature_acts, fwd.active_masks, fwd.errors_detached):
        for k, v in layer_dict.items():
            if v.dim() == 3:
                layer_dict[k] = v[0]
    for d in (fwd.error_scales,):
        for k, v in d.items():
            if v.dim() == 3:
                d[k] = v[0]
    if fwd.embeddings_detached is not None and fwd.embeddings_detached.dim() == 3:
        fwd.embeddings_detached = fwd.embeddings_detached[0]
    if fwd.embed_scale is not None and fwd.embed_scale.dim() == 3:
        fwd.embed_scale = fwd.embed_scale[0]

    return fwd


# ---------------------------------------------------------------------------
# Gradient-enabled forward for feature-to-feature edges
# ---------------------------------------------------------------------------


def _linearized_forward_with_residual_leaves(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    caps: _CapturedConstants,
    *,
    n_bos_tokens: int = 1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
    embed_module_name: str = "model.embed_tokens",
) -> tuple[Tensor, dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]]:
    """Run the linearized model with residual streams as leaves at each MLP input.

    Returns:
        ``(logits, residual_leaves, feature_acts_detached, active_masks)``

    ``residual_leaves[layer]`` is the post-attention-layernorm input to the
    transcoder at each layer, stored as a leaf with ``requires_grad``.
    Gradients through these leaves enable feature-to-feature virtual weight
    computation.
    """

    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    handles: list[torch.utils.hooks.RemovableHook] = []
    saved_forwards: dict[int, Any] = {}
    residual_leaves: dict[int, Tensor] = {}
    feature_acts_detached: dict[int, Tensor] = {}
    active_masks: dict[int, Tensor] = {}
    buf = _CrossLayerBuffer() if not is_set else None

    def _make_residual_leaf_plt_hook(layer_idx: int, tc: TranscoderSet) -> Any:
        def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
            x = inp[0]
            original_out = output[0] if isinstance(output, tuple) else output

            # Make the transcoder input a leaf
            x_leaf = x.detach().clone().requires_grad_(True)
            residual_leaves[layer_idx] = x_leaf

            # Normal transcoder forward from the leaf
            tc_layer = tc.transcoders[layer_idx]
            pre_acts = F.linear(x_leaf.to(tc_layer.W_enc.dtype), tc_layer.W_enc, tc_layer.b_enc)
            acts = tc_layer.activation_function(pre_acts)
            feature_acts_detached[layer_idx] = acts.detach()
            active_masks[layer_idx] = acts.detach() > 0

            reconstruction = acts @ tc_layer.W_dec + tc_layer.b_dec
            if tc_layer.W_skip is not None:
                reconstruction = reconstruction + x_leaf @ tc_layer.W_skip.T

            if n_bos_tokens > 0:
                reconstruction = torch.cat(
                    [
                        original_out[..., :n_bos_tokens, :].detach(),
                        reconstruction[..., n_bos_tokens:, :],
                    ],
                    dim=-2,
                )

            # Add error (constant)
            recon_det = tc_layer(x.detach())
            if n_bos_tokens > 0:
                recon_det = torch.cat(
                    [
                        original_out[..., :n_bos_tokens, :].detach(),
                        recon_det[..., n_bos_tokens:, :],
                    ],
                    dim=-2,
                )
            error = caps.mlp_outputs[layer_idx] - recon_det.detach()
            result = reconstruction + error.detach()
            return _replace_output(output, result)

        return hook

    def _make_residual_leaf_clt_hook(
        layer_idx: int, clt_: CrossLayerTranscoder, buf_: _CrossLayerBuffer
    ) -> Any:
        def hook(_mod: nn.Module, inp: tuple[Any, ...], output: Any) -> Any:
            x = inp[0]
            original_out = output[0] if isinstance(output, tuple) else output

            x_leaf = x.detach().clone().requires_grad_(True)
            residual_leaves[layer_idx] = x_leaf

            # Encode from leaf
            pre_acts = clt_.encode_layer(
                x_leaf.to(clt_.W_enc.dtype), layer_idx, apply_activation_function=False
            )
            acts = clt_.apply_activation_function(layer_idx, pre_acts)
            feature_acts_detached[layer_idx] = acts.detach()
            active_masks[layer_idx] = acts.detach() > 0

            # Decode
            W_dec = clt_._get_decoder_vectors(layer_idx)
            self_contrib = torch.einsum("...f,fd->...d", acts, W_dec[:, 0, :])
            for offset in range(1, W_dec.shape[1]):
                future_contrib = torch.einsum("...f,fd->...d", acts, W_dec[:, offset, :])
                buf_.add(layer_idx + offset, future_contrib)

            reconstruction = clt_.b_dec[layer_idx] + self_contrib
            buffered = buf_.pop(layer_idx)
            if buffered is not None:
                reconstruction = reconstruction + buffered
            if clt_.skip_connection:
                reconstruction = reconstruction + clt_.compute_skip(layer_idx, x_leaf)

            if n_bos_tokens > 0:
                reconstruction = torch.cat(
                    [
                        original_out[..., :n_bos_tokens, :].detach(),
                        reconstruction[..., n_bos_tokens:, :],
                    ],
                    dim=-2,
                )

            # Error (constant)
            from llm_circuits.circuits.replacement_model import _compute_clt_reconstruction

            err_buf = _CrossLayerBuffer()
            recon_det = _compute_clt_reconstruction(x.detach(), layer_idx, clt_, err_buf)
            if n_bos_tokens > 0:
                recon_det = torch.cat(
                    [
                        original_out[..., :n_bos_tokens, :].detach(),
                        recon_det[..., n_bos_tokens:, :],
                    ],
                    dim=-2,
                )
            error = caps.mlp_outputs[layer_idx] - recon_det.detach()
            result = reconstruction + error.detach()
            return _replace_output(output, result)

        return hook

    try:
        # Frozen norms
        for tmpl in layernorm_templates:
            for i in range(n_layers):
                name = tmpl.format(layer=i)
                mod = model.get_submodule(name)
                handles.append(
                    mod.register_forward_hook(_make_frozen_rmsnorm_hook(name, caps.rmsnorm_scales))
                )
        final_mod = model.get_submodule(final_norm_name)
        handles.append(
            final_mod.register_forward_hook(
                _make_frozen_rmsnorm_hook(final_norm_name, caps.rmsnorm_scales)
            )
        )

        # Frozen attention
        for i in range(n_layers):
            attn_mod = model.get_submodule(attn_name_template.format(layer=i))
            saved_forwards[i] = attn_mod.forward
            attn_mod.forward = _make_frozen_attn_forward(attn_mod, caps.attn_weights[i])

        # Residual-leaf transcoder hooks
        for i in range(n_layers):
            mlp_mod = model.get_submodule(mlp_name_template.format(layer=i))
            if is_set:
                handles.append(
                    mlp_mod.register_forward_hook(_make_residual_leaf_plt_hook(i, transcoder))
                )
            else:
                handles.append(
                    mlp_mod.register_forward_hook(_make_residual_leaf_clt_hook(i, transcoder, buf))
                )

        logits = model(input_ids).logits

    finally:
        for h in handles:
            h.remove()
        for i, orig_fwd in saved_forwards.items():
            attn_mod = model.get_submodule(attn_name_template.format(layer=i))
            attn_mod.forward = orig_fwd
        if buf is not None:
            buf.clear()

    # Squeeze batch dim
    if logits.dim() == 3:
        logits = logits[0]
    for d in (residual_leaves, feature_acts_detached, active_masks):
        for k, v in d.items():
            if v.dim() == 3:
                d[k] = v[0]

    return logits, residual_leaves, feature_acts_detached, active_masks


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------


def _collect_active_feature_nodes(
    fwd_or_acts: dict[int, Tensor],
    masks: dict[int, Tensor],
    n_layers: int,
    activation_threshold: float,
) -> list[tuple[int, int, int, float]]:
    """Return ``(layer, position, feature_idx, activation)`` for active features."""
    nodes: list[tuple[int, int, int, float]] = []
    for layer in range(n_layers):
        acts = fwd_or_acts[layer]  # (seq, d_tc)
        mask = masks[layer]  # (seq, d_tc)
        positions, feat_idxs = torch.where(mask)
        for p, f in zip(positions.tolist(), feat_idxs.tolist(), strict=True):
            act_val = acts[p, f].item()
            if abs(act_val) > activation_threshold:
                nodes.append((layer, p, f, act_val))
    return nodes


def _compute_logit_edges(
    fwd: _ForwardResult,
    *,
    target_positions: list[int],
    target_token_ids: Tensor,
    n_layers: int,
    edge_threshold: float,
    activation_threshold: float,
) -> tuple[list[Node], list[tuple[NodeId, NodeId, float]]]:
    """Compute edges from all source nodes to logit target nodes."""
    nodes: list[Node] = []
    edges: list[tuple[NodeId, NodeId, float]] = []

    # Collect all leaves for a single autograd.grad call
    all_leaves: list[Tensor] = []
    # feature_acts for each layer
    for layer in range(n_layers):
        all_leaves.append(fwd.feature_acts[layer])
    # error_scales for each layer
    for layer in range(n_layers):
        all_leaves.append(fwd.error_scales[layer])
    # embed_scale
    all_leaves.append(fwd.embed_scale)

    seq_len = fwd.logits.shape[0]

    for pos in target_positions:
        tok_id = (
            target_token_ids[pos].item() if target_token_ids.dim() > 0 else target_token_ids.item()
        )
        logit_val = fwd.logits[pos, tok_id]

        target_nid = NodeId(NodeType.LOGIT, n_layers, pos, tok_id)
        nodes.append(Node(id=target_nid, activation=logit_val.item()))

        grads = torch.autograd.grad(
            logit_val,
            all_leaves,
            retain_graph=True,
            allow_unused=True,
        )

        # Feature edges
        for layer in range(n_layers):
            g = grads[layer]  # (seq, d_tc) or None
            if g is None:
                continue
            acts = fwd.feature_acts[layer]
            mask = fwd.active_masks[layer]
            positions, feat_idxs = torch.where(mask)
            for p, f in zip(positions.tolist(), feat_idxs.tolist(), strict=True):
                act_val = acts[p, f].item()
                if abs(act_val) <= activation_threshold:
                    continue
                virtual_w = g[p, f].item()
                edge_w = act_val * virtual_w
                if abs(edge_w) > edge_threshold:
                    src = NodeId(NodeType.FEATURE, layer, p, f)
                    edges.append((src, target_nid, edge_w))

        # Error edges
        for layer in range(n_layers):
            g = grads[n_layers + layer]  # (seq, 1) or None
            if g is None:
                continue
            for p in range(seq_len):
                edge_w = g[p, 0].item()
                if abs(edge_w) > edge_threshold:
                    src = NodeId(NodeType.ERROR, layer, p, -1)
                    edges.append((src, target_nid, edge_w))

        # Embedding edges
        g_embed = grads[-1]  # (seq, 1) or None
        if g_embed is not None:
            for p in range(seq_len):
                edge_w = g_embed[p, 0].item()
                if abs(edge_w) > edge_threshold:
                    src = NodeId(NodeType.EMBEDDING, -1, p, -1)
                    edges.append((src, target_nid, edge_w))

    return nodes, edges


def _compute_feature_to_feature_edges(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    caps: _CapturedConstants,
    active_features: list[tuple[int, int, int, float]],
    *,
    n_bos_tokens: int,
    n_layers: int,
    edge_threshold: float,
    mlp_name_template: str,
    attn_name_template: str,
    layernorm_templates: list[str] | None,
    final_norm_name: str,
    embed_module_name: str,
) -> list[tuple[NodeId, NodeId, float]]:
    """Compute feature-to-feature edges using residual-leaf forward pass."""
    is_set = _is_transcoder_set(transcoder)
    edges: list[tuple[NodeId, NodeId, float]] = []

    _logits, residual_leaves, feat_acts_det, act_masks = _linearized_forward_with_residual_leaves(
        model,
        transcoder,
        input_ids,
        caps,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
        embed_module_name=embed_module_name,
    )

    # For each target feature, backward through residual leaves, then dot with
    # source decoder vectors to get virtual weights.
    # Group targets by layer for potential batching.
    target_features = active_features  # all active features are targets

    for tgt_layer, tgt_pos, tgt_feat, _tgt_act in target_features:
        if tgt_layer == 0:
            continue  # no upstream features

        # Get the target feature's encoder weight
        if is_set:
            tc = transcoder.transcoders[tgt_layer]
            W_enc_g = tc.W_enc[tgt_feat]  # (d_model,)
        else:
            W_enc_g = transcoder.W_enc[tgt_layer, tgt_feat]  # (d_model,)

        # The target feature's pre-activation is W_enc_g @ residual_input_to_transcoder
        # residual_leaves[tgt_layer] is the input to the transcoder at tgt_layer
        residual_at_tgt = residual_leaves[tgt_layer]  # (seq, d_model)
        pre_act_g = residual_at_tgt[tgt_pos] @ W_enc_g  # scalar

        # Gradient w.r.t. all upstream residual leaves
        upstream_leaves = [residual_leaves[li] for li in range(tgt_layer)]
        if not upstream_leaves:
            continue

        grads = torch.autograd.grad(
            pre_act_g,
            upstream_leaves,
            retain_graph=True,
            allow_unused=True,
        )

        target_nid = NodeId(NodeType.FEATURE, tgt_layer, tgt_pos, tgt_feat)

        for src_layer_offset, g in enumerate(grads):
            src_layer = src_layer_offset
            if g is None:
                continue

            # Virtual weight from source feature f: grad @ W_dec_f
            src_mask = act_masks[src_layer]  # (seq, d_tc)
            positions, feat_idxs = torch.where(src_mask)

            if len(positions) == 0:
                continue

            # Get decoder weights for active features
            if is_set:
                src_tc = transcoder.transcoders[src_layer]
                W_dec_active = src_tc.W_dec[feat_idxs]  # (n_active, d_model)
            else:
                W_dec_all = transcoder._get_decoder_vectors(src_layer)
                # For CLT: use offset 0 decoder (self-layer contribution direction)
                # The grad already captures multi-layer propagation
                W_dec_active = W_dec_all[feat_idxs, 0, :]  # (n_active, d_model)

            # Compute virtual weights for all active source features at this layer
            for idx, (p, f) in enumerate(zip(positions.tolist(), feat_idxs.tolist(), strict=True)):
                src_act = feat_acts_det[src_layer][p, f].item()
                vw = g[p] @ W_dec_active[idx]  # scalar
                edge_w = src_act * vw.item()
                if abs(edge_w) > edge_threshold:
                    src_nid = NodeId(NodeType.FEATURE, src_layer, p, f)
                    edges.append((src_nid, target_nid, edge_w))

    return edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_attribution_graph(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    tokenizer: Any,
    *,
    target_positions: list[int] | None = None,
    target_token_ids: Tensor | None = None,
    feature_to_feature: bool = False,
    activation_threshold: float = 0.0,
    edge_threshold: float = 0.0,
    n_bos_tokens: int = 1,
    model_name: str = "",
    transcoder_repo: str = "",
    mlp_name_template: str = "model.layers.{layer}.mlp",
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
    lm_head_name: str = "lm_head",
    embed_module_name: str = "model.embed_tokens",
) -> AttributionGraph:
    """Build an attribution graph for the given input.

    Performs a fully linearized forward pass (frozen attention, frozen RMSNorm,
    transcoders with error nodes) and computes edge weights via autograd
    backward passes.

    Args:
        model: The full language model.
        transcoder: A ``TranscoderSet`` or ``CrossLayerTranscoder``.
        input_ids: Token ids, shape ``(batch, seq)`` or ``(seq,)``.
        tokenizer: Tokenizer for decoding token strings.
        target_positions: Sequence positions to trace. Defaults to ``[-1]``.
        target_token_ids: Token ids for each target position. Defaults to
            argmax of the model output at each target position.
        feature_to_feature: If ``True``, also compute edges between every
            pair of active features (slower, richer graph).
        activation_threshold: Minimum ``|activation|`` for a feature node.
        edge_threshold: Minimum ``|edge weight|`` to include.
        n_bos_tokens: Number of leading positions to preserve.
        model_name: Stored in graph metadata.
        transcoder_repo: Stored in graph metadata.

    Returns:
        An :class:`AttributionGraph`.
    """
    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    # Ensure 2-d input
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    seq_len = input_ids.shape[1]
    all_token_ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([t]) for t in all_token_ids]

    # Default target: last position
    if target_positions is None:
        target_positions = [seq_len - 1]

    # ------------------------------------------------------------------
    # Phase 1: Capture constants
    # ------------------------------------------------------------------
    log.info("Phase 1: capturing constants from original forward pass")
    caps = _capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
        freeze_attention=True,
        freeze_layernorms=True,
    )

    # ------------------------------------------------------------------
    # Phase 2: Forward with feature leaves (for logit edges)
    # ------------------------------------------------------------------
    log.info("Phase 2: gradient-enabled linearized forward pass")
    fwd = _linearized_forward_with_features(
        model,
        transcoder,
        input_ids,
        caps,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
        embed_module_name=embed_module_name,
    )

    # Default target tokens: argmax
    if target_token_ids is None:
        target_token_ids = fwd.logits.argmax(dim=-1)  # (seq,)

    # ------------------------------------------------------------------
    # Phase 3: Compute edges
    # ------------------------------------------------------------------
    log.info("Phase 3: computing edges via backward passes")

    # 3a. Logit edges
    logit_nodes, logit_edges = _compute_logit_edges(
        fwd,
        target_positions=target_positions,
        target_token_ids=target_token_ids,
        n_layers=n_layers,
        edge_threshold=edge_threshold,
        activation_threshold=activation_threshold,
    )

    # ------------------------------------------------------------------
    # Build node set
    # ------------------------------------------------------------------
    graph = AttributionGraph(
        tokens=tokens,
        input_ids=all_token_ids,
        model_name=model_name,
        transcoder_repo=transcoder_repo,
    )

    # Add logit target nodes
    for n in logit_nodes:
        graph.nodes[n.id] = n

    # Add feature nodes (all active features)
    active_features = _collect_active_feature_nodes(
        fwd.feature_acts,
        fwd.active_masks,
        n_layers,
        activation_threshold,
    )
    for layer, pos, feat, act in active_features:
        nid = NodeId(NodeType.FEATURE, layer, pos, feat)
        if nid not in graph.nodes:
            graph.nodes[nid] = Node(id=nid, activation=act)

    # Add error nodes
    for layer in range(n_layers):
        err = fwd.errors_detached[layer]  # (seq, d_model)
        for p in range(seq_len):
            act = err[p].norm().item()
            if act > 0:
                nid = NodeId(NodeType.ERROR, layer, p, -1)
                graph.nodes[nid] = Node(id=nid, activation=act)

    # Add embedding nodes
    if fwd.embeddings_detached is not None:
        for p in range(seq_len):
            act = fwd.embeddings_detached[p].norm().item()
            nid = NodeId(NodeType.EMBEDDING, -1, p, -1)
            graph.nodes[nid] = Node(id=nid, activation=act)

    # Add logit edges
    graph.edges.extend(logit_edges)

    # ------------------------------------------------------------------
    # 3b. Feature-to-feature edges (optional)
    # ------------------------------------------------------------------
    if feature_to_feature:
        log.info("Phase 3b: computing feature-to-feature edges")
        f2f_edges = _compute_feature_to_feature_edges(
            model,
            transcoder,
            input_ids,
            caps,
            active_features,
            n_bos_tokens=n_bos_tokens,
            n_layers=n_layers,
            edge_threshold=edge_threshold,
            mlp_name_template=mlp_name_template,
            attn_name_template=attn_name_template,
            layernorm_templates=layernorm_templates,
            final_norm_name=final_norm_name,
            embed_module_name=embed_module_name,
        )
        graph.edges.extend(f2f_edges)

    # Ensure source nodes referenced in edges exist in the graph
    for src, _tgt, _w in graph.edges:
        if src not in graph.nodes:
            graph.nodes[src] = Node(id=src, activation=0.0)

    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    log.info(f"Attribution graph built: {n_nodes} nodes, {n_edges} edges")

    return graph
