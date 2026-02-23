"""Attribution graph construction via autograd on the local replacement model.

Edges represent the influence of upstream nodes (embedding, transcoder features,
errors) on downstream nodes (transcoder pre-activations, logits), computed by
back-propagating through the linearised residual stream using
:func:`torch.autograd.grad`.

The main entry point is :func:`build_attribution_graph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from llm_circuits.circuits.local_replacement_model import (
    LocalReplacementModel,
    capture_constants,
)
from llm_circuits.logging import get_logger

if TYPE_CHECKING:
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
    from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default templates (Qwen3)
# ---------------------------------------------------------------------------

_QWEN3_LAYERNORM_TEMPLATES: list[str] = [
    "model.layers.{layer}.input_layernorm",
    "model.layers.{layer}.post_attention_layernorm",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AttributionNode:
    """A node in the attribution graph."""

    node_type: str
    """One of ``"embedding"``, ``"feature"``, ``"error"``, ``"logit"``."""

    layer: int
    """Layer index.  -1 for embedding, *n_layers* for logit."""

    position: int
    """Sequence position."""

    feature_idx: int | None = None
    """Transcoder feature index (only for ``"feature"`` nodes)."""

    token_id: int | None = None
    """Vocabulary token id (only for ``"logit"`` nodes)."""

    activation: float = 0.0
    """Scalar activation value (feature activation, logit value, or L2 norm)."""


@dataclass
class AttributionEdge:
    """A weighted directed edge in the attribution graph."""

    source: int
    """Index into :attr:`AttributionGraph.nodes`."""

    target: int
    """Index into :attr:`AttributionGraph.nodes`."""

    weight: float
    """Edge weight (dot product of gradient and contribution vector)."""


@dataclass
class AttributionGraph:
    """Attribution graph: nodes and weighted edges."""

    nodes: list[AttributionNode] = field(default_factory=list)
    edges: list[AttributionEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_transcoder_set(tc: Any) -> bool:
    from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet

    return isinstance(tc, TranscoderSet)


def _get_source_contribution(
    node: AttributionNode,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    features: dict[int, Tensor],
    errors: dict[int, Tensor],
    embedding: Tensor,
    is_set: bool,
) -> Tensor:
    """Return the contribution vector for a source node.

    * Feature at (l, q, f): ``activation * W_dec[f, :]``
    * Error at (l, q): ``errors[l][q, :]``
    * Embedding at q: ``embedding[q, :]``
    """
    if node.node_type == "feature":
        layer = node.layer
        feat_idx = node.feature_idx
        act = features[layer]
        # Get the scalar activation
        act_val = act[node.position, feat_idx]
        if is_set:
            dec_vec = transcoder.transcoders[layer].W_dec[feat_idx, :]
        else:
            # CLT: W_dec shape is (d_transcoder, n_target_layers, d_model)
            # Self-contribution is offset 0
            dec_vec = transcoder._get_decoder_vectors(layer)[feat_idx, 0, :]
        return act_val * dec_vec

    if node.node_type == "error":
        return errors[node.layer][node.position, :]

    if node.node_type == "embedding":
        return embedding[node.position, :]

    raise ValueError(f"Cannot compute contribution for node type {node.node_type!r}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_attribution_graph(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    *,
    n_bos_tokens: int = 1,
    top_k_logits: int = 3,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
    decoder_layer_template: str = "model.layers.{layer}",
    embed_module_name: str = "model.embed_tokens",
) -> AttributionGraph:
    """Build an attribution graph for *input_ids*.

    Performs three forward passes:

    1. **Capture pass** — records attention weights, RMSNorm scales, MLP outputs.
    2. **Local replacement pass** — with ``capture_pre_activations=True`` and a
       gradient seed on the embedding layer so that ``autograd.grad`` can trace
       influence through the linearised residual stream.
    3. **Edge computation** — for every target node (feature pre-activation or
       logit), compute ``autograd.grad`` w.r.t. the embedding and every
       intermediate residual, then dot-product with each source's contribution
       vector.
    """
    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    # Ensure input_ids has batch dim
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    seq_len = input_ids.shape[1]

    # ------------------------------------------------------------------
    # Phase 1: capture constants
    # ------------------------------------------------------------------
    caps = capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )

    # ------------------------------------------------------------------
    # Phase 2: local replacement forward with gradient seed
    # ------------------------------------------------------------------
    # We need to capture:
    # - embedding tensor (with requires_grad)
    # - residual stream at each decoder layer output (in the autograd graph)
    # - pre_activations (non-detached)
    # - logits (non-detached)

    embedding_ref: list[Tensor] = []  # will hold the seeded embedding
    residuals: dict[int, Tensor] = {}  # layer_idx -> residual tensor

    # Gradient seed hook: detach embedding output and set requires_grad
    def _embed_hook(_mod: nn.Module, _inp: Any, output: Tensor) -> Tensor:
        seeded = output.detach().requires_grad_(True)
        embedding_ref.append(seeded)
        return seeded

    # Residual capture hook on each decoder layer
    def _make_residual_hook(layer_idx: int) -> Any:
        def hook(_mod: nn.Module, _inp: Any, output: Any) -> None:
            # Decoder layer returns (hidden_states, ...) or just hidden_states
            res = output[0] if isinstance(output, tuple) else output
            residuals[layer_idx] = res

        return hook

    embed_mod = model.get_submodule(embed_module_name)
    extra_handles: list[torch.utils.hooks.RemovableHook] = []
    extra_handles.append(embed_mod.register_forward_hook(_embed_hook))

    for i in range(n_layers):
        dec_mod = model.get_submodule(decoder_layer_template.format(layer=i))
        extra_handles.append(dec_mod.register_forward_hook(_make_residual_hook(i)))

    try:
        with LocalReplacementModel(
            model,
            transcoder,
            caps,
            include_error=True,
            n_bos_tokens=n_bos_tokens,
            mlp_name_template=mlp_name_template,
            output_module_template=output_module_template,
            attn_name_template=attn_name_template,
            layernorm_templates=layernorm_templates,
            final_norm_name=final_norm_name,
            capture_pre_activations=True,
        ) as local_model:
            ctx = local_model.forward(input_ids)
    finally:
        for h in extra_handles:
            h.remove()

    # Unpack
    embedding = embedding_ref[0]  # (1, seq, d_model)
    logits = ctx.logits  # (seq, vocab) — already squeezed
    features = ctx.features  # dict[layer, Tensor(seq, d_transcoder)] — detached
    errors = ctx.errors  # dict[layer, Tensor(seq, d_model)] — detached
    pre_activations = ctx.pre_activations  # dict[layer, Tensor(1, seq, d_transcoder)] — in graph

    # Squeeze batch dim from embedding for contribution vectors
    embed_squeezed = embedding[0] if embedding.dim() == 3 else embedding  # (seq, d_model)

    # Squeeze batch dim from errors if needed
    for k, v in errors.items():
        if v.dim() == 3:
            errors[k] = v[0]
    # features are already squeezed by LocalReplacementModel.forward
    # but let's be safe
    for k, v in features.items():
        if v.dim() == 3:
            features[k] = v[0]

    # ------------------------------------------------------------------
    # Phase 3: build node list
    # ------------------------------------------------------------------
    graph = AttributionGraph()
    node_index: dict[tuple, int] = {}  # (type, layer, pos, feat/token) -> node idx

    # Embedding nodes: one per position
    for p in range(seq_len):
        idx = len(graph.nodes)
        node_index[("embedding", -1, p, None)] = idx
        graph.nodes.append(
            AttributionNode(
                node_type="embedding",
                layer=-1,
                position=p,
                activation=embed_squeezed[p].norm().item(),
            )
        )

    # Feature nodes: non-zero activations per (layer, position)
    for layer in sorted(features.keys()):
        feat_tensor = features[layer]  # (seq, d_transcoder)
        for p in range(n_bos_tokens, seq_len):
            nonzero_feats = feat_tensor[p].nonzero(as_tuple=True)[0]
            for f_idx in nonzero_feats:
                f = f_idx.item()
                idx = len(graph.nodes)
                node_index[("feature", layer, p, f)] = idx
                graph.nodes.append(
                    AttributionNode(
                        node_type="feature",
                        layer=layer,
                        position=p,
                        feature_idx=f,
                        activation=feat_tensor[p, f].item(),
                    )
                )

    # Error nodes: one per (layer, position)
    for layer in sorted(errors.keys()):
        for p in range(n_bos_tokens, seq_len):
            idx = len(graph.nodes)
            node_index[("error", layer, p, None)] = idx
            graph.nodes.append(
                AttributionNode(
                    node_type="error",
                    layer=layer,
                    position=p,
                    activation=errors[layer][p].norm().item(),
                )
            )

    # Logit nodes: top-k at the last position
    last_pos = seq_len - 1
    logit_vals, logit_ids = logits[last_pos].topk(top_k_logits)
    for rank in range(top_k_logits):
        tok = logit_ids[rank].item()
        idx = len(graph.nodes)
        node_index[("logit", n_layers, last_pos, tok)] = idx
        graph.nodes.append(
            AttributionNode(
                node_type="logit",
                layer=n_layers,
                position=last_pos,
                token_id=tok,
                activation=logit_vals[rank].item(),
            )
        )

    # ------------------------------------------------------------------
    # Phase 4: compute edges via autograd.grad
    # ------------------------------------------------------------------
    # Build list of tensors to differentiate w.r.t.:
    # [embedding, residuals[0], residuals[1], ..., residuals[n_layers-1]]
    # Index 0 = embedding, index l+1 = residuals[l]

    def _compute_edges_for_target(
        target_scalar: Tensor,
        target_node_idx: int,
        source_layers: list[int],
    ) -> None:
        """Compute edges from all sources at *source_layers* to the target."""
        grad_inputs = [embedding]
        for sl in source_layers:
            grad_inputs.append(residuals[sl])

        grads = torch.autograd.grad(
            target_scalar,
            grad_inputs,
            retain_graph=True,
            allow_unused=True,
        )

        # grads[0] = d(target)/d(embedding), shape (1, seq, d_model)
        # grads[k+1] = d(target)/d(residuals[source_layers[k]]), shape (1, seq, d_model)

        # Iterate over all source nodes and compute edge weights
        for node_idx, node in enumerate(graph.nodes):
            if node.node_type not in ("embedding", "feature", "error"):
                continue

            # Determine which grad tensor and position to use
            if node.node_type == "embedding":
                grad_idx = 0
                grad_layer = -1
            else:
                grad_layer = node.layer
                if grad_layer not in source_layers:
                    continue
                grad_idx = source_layers.index(grad_layer) + 1

            grad_tensor = grads[grad_idx]
            if grad_tensor is None:
                continue

            # Get contribution vector
            contrib = _get_source_contribution(
                node, transcoder, features, errors, embed_squeezed, is_set
            )

            # Grad shape is (1, seq, d_model) or (seq, d_model)
            if grad_tensor.dim() == 3:
                g = grad_tensor[0, node.position, :]
            else:
                g = grad_tensor[node.position, :]

            weight = (g @ contrib).item()
            if weight != 0.0:
                graph.edges.append(
                    AttributionEdge(source=node_idx, target=target_node_idx, weight=weight)
                )

    # --- Feature targets ---
    for node_idx, node in enumerate(graph.nodes):
        if node.node_type != "feature":
            continue
        layer = node.layer
        pre_act = pre_activations[layer]
        # pre_act shape: (1, seq, d_transcoder) or (seq, d_transcoder)
        if pre_act.dim() == 3:
            target_scalar = pre_act[0, node.position, node.feature_idx]
        else:
            target_scalar = pre_act[node.position, node.feature_idx]

        # Sources can come from embedding + all layers before this one
        source_layers = list(range(layer))
        _compute_edges_for_target(target_scalar, node_idx, source_layers)

    # --- Logit targets ---
    # logits is (seq, vocab), but we need it connected to the autograd graph.
    # ctx.logits was computed inside the local model forward, so it should be in the graph
    # because embedding has requires_grad=True.
    for node_idx, node in enumerate(graph.nodes):
        if node.node_type != "logit":
            continue
        target_scalar = logits[node.position, node.token_id]
        source_layers = list(range(n_layers))
        _compute_edges_for_target(target_scalar, node_idx, source_layers)

    return graph
