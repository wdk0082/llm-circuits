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
    _QWEN3_LAYERNORM_TEMPLATES,
    LocalReplacementModel,
    capture_constants,
)
from llm_circuits.logging import get_logger

if TYPE_CHECKING:
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
    from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet

log = get_logger(__name__)

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

    label: dict | None = None
    """Feature label metadata (only for ``"feature"`` nodes).

    When populated, contains ``top_logits``, ``bottom_logits``, and
    ``activation_frequency`` from the transcoder repo's feature index.
    """


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


def _batched_edge_weights(
    grad_2d: Tensor,
    positions: Tensor,
    contrib_mat: Tensor,
) -> Tensor:
    """Edge weights for a batch of sources sharing a gradient tensor.

    ``weight_i = <grad_2d[positions_i], contrib_i>``.  ``grad_2d`` is
    ``(seq, d_model)``, ``positions`` is ``(n,)`` long, ``contrib_mat`` is
    ``(n, d_model)``.  Returns ``(n,)``.  This is the fused replacement for the
    old per-source ``(grad[pos] @ contrib).item()`` loop — one device->host
    transfer instead of one per edge.
    """
    g = grad_2d.index_select(0, positions).to(contrib_mat.dtype)  # (n, d_model)
    return (g * contrib_mat).sum(dim=1)


def _batched_edge_weights_multi(
    grad_3d: Tensor,
    positions: Tensor,
    contrib_mat: Tensor,
) -> Tensor:
    """Edge weights for a batch of TARGETS against a batch of sources.

    ``grad_3d`` is ``(B_targets, seq, d_model)`` (per-target gradient w.r.t. one
    source layer's residual), ``positions`` is ``(n_src,)`` long, ``contrib_mat`` is
    ``(n_src, d_model)``.  Returns ``(B_targets, n_src)`` where
    ``w[b, s] = <grad_3d[b, positions[s]], contrib_mat[s]>`` — the multi-target
    generalisation of :func:`_batched_edge_weights`.
    """
    g = grad_3d.index_select(1, positions).to(contrib_mat.dtype)  # (B, n_src, d_model)
    return (g * contrib_mat.unsqueeze(0)).sum(dim=2)  # (B, n_src)


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
    max_feature_targets: int | None = None,
    max_feature_nodes: int | None = None,
    edge_batch_size: int = 128,
    min_edge_weight: float = 0.0,
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

    # Feature nodes: non-zero activations per (layer, position).  Optionally cap to the
    # top-`max_feature_nodes` by |activation| so dense inputs (which can fire ~1M+
    # features) stay bounded for edge computation and the prune adjacency matrix.
    # Default None keeps every non-zero feature (unchanged for the existing pipelines).
    feat_candidates: list[tuple[int, int, int, float]] = []  # (layer, position, feat, act)
    for layer in sorted(features.keys()):
        feat_tensor = features[layer]  # (seq, d_transcoder)
        for p in range(n_bos_tokens, seq_len):
            row = feat_tensor[p]
            nz = row.nonzero(as_tuple=True)[0]
            if nz.numel() == 0:
                continue
            # vectorised read (avoids a GPU->CPU sync per feature)
            for f, act in zip(nz.tolist(), row[nz].tolist(), strict=True):
                feat_candidates.append((layer, p, f, act))
    if max_feature_nodes is not None and len(feat_candidates) > max_feature_nodes:
        feat_candidates.sort(key=lambda c: abs(c[3]), reverse=True)
        feat_candidates = feat_candidates[:max_feature_nodes]
    feat_candidates.sort(key=lambda c: (c[0], c[1], c[2]))  # stable node ordering
    for layer, p, f, act in feat_candidates:
        idx = len(graph.nodes)
        node_index[("feature", layer, p, f)] = idx
        graph.nodes.append(
            AttributionNode(
                node_type="feature", layer=layer, position=p, feature_idx=f, activation=act
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

    log.info(
        "Built %d nodes (emb=%d, feat=%d, err=%d, logit=%d)",
        len(graph.nodes),
        sum(1 for n in graph.nodes if n.node_type == "embedding"),
        sum(1 for n in graph.nodes if n.node_type == "feature"),
        sum(1 for n in graph.nodes if n.node_type == "error"),
        sum(1 for n in graph.nodes if n.node_type == "logit"),
    )

    # ------------------------------------------------------------------
    # Phase 4: pre-compute source contribution vectors
    # ------------------------------------------------------------------
    # Build them once so we don't hit lazy-loaded W_dec per edge.
    # source_contribs[node_idx] = contribution vector (d_model,)
    # source_layer_map[layer] = list of (node_idx, position, contrib) for that layer
    # embed_sources = list of (node_idx, position, contrib) for embedding nodes

    log.info("Pre-computing source contribution vectors ...")
    dev = embed_squeezed.device
    source_by_layer: dict[int, list[tuple[int, int, Tensor]]] = {lay: [] for lay in range(n_layers)}
    embed_sources: list[tuple[int, int, Tensor]] = []

    # Embedding sources
    for node_idx, node in enumerate(graph.nodes):
        if node.node_type == "embedding":
            contrib = embed_squeezed[node.position, :]
            embed_sources.append((node_idx, node.position, contrib))

    # Error sources (no W_dec needed)
    for node_idx, node in enumerate(graph.nodes):
        if node.node_type == "error":
            contrib = errors[node.layer][node.position, :]
            source_by_layer[node.layer].append((node_idx, node.position, contrib))

    # Feature sources — batch W_dec reads per layer to avoid repeated lazy loads
    feat_nodes_by_layer: dict[int, list[tuple[int, AttributionNode]]] = {}
    for node_idx, node in enumerate(graph.nodes):
        if node.node_type == "feature":
            feat_nodes_by_layer.setdefault(node.layer, []).append((node_idx, node))

    for layer in sorted(feat_nodes_by_layer):
        layer_nodes = feat_nodes_by_layer[layer]
        feat_indices = torch.tensor([n.feature_idx for _, n in layer_nodes], dtype=torch.long)
        if is_set:
            # Load W_dec once for this layer (handles lazy loading internally)
            dec_vecs = transcoder.transcoders[layer]._get_decoder_vectors(feat_indices)
        else:
            # CLT: shape (n_feats, n_target_layers, d_model); take offset 0 (self-layer)
            dec_vecs = transcoder._get_decoder_vectors(layer, feat_indices)[:, 0, :]

        for k, (node_idx, node) in enumerate(layer_nodes):
            act_val = features[node.layer][node.position, node.feature_idx]
            contrib = act_val * dec_vecs[k]
            source_by_layer[node.layer].append((node_idx, node.position, contrib))
        log.info("  layer %d: %d feature contributions computed", layer, len(layer_nodes))

    # Stack each source group into (n_sources, d_model) matrices once so that each
    # (target, source-layer) edge computation is a single fused index-select + dot,
    # rather than one GPU->CPU ``.item()`` sync per edge (the old hot path).
    def _to_batch(
        rows: list[tuple[int, int, Tensor]],
    ) -> tuple[list[int], Tensor, Tensor] | None:
        if not rows:
            return None
        node_idx = [r[0] for r in rows]
        positions = torch.tensor([r[1] for r in rows], dtype=torch.long, device=dev)
        contrib_mat = torch.stack([r[2] for r in rows]).float()  # (n_sources, d_model)
        return node_idx, positions, contrib_mat

    embed_batch = _to_batch(embed_sources)
    layer_batches: dict[int, tuple[list[int], Tensor, Tensor]] = {}
    for lay in range(n_layers):
        batch = _to_batch(source_by_layer[lay])
        if batch is not None:
            layer_batches[lay] = batch

    # ------------------------------------------------------------------
    # Phase 5: compute edges via BATCHED autograd.  Targets that share a source
    # set (same layer) are differentiated together with one vmapped backward
    # (``is_grads_batched``), chunked to ``edge_batch_size`` for memory — i.e.
    # ~one backward per (layer, chunk) instead of one per target.
    # ------------------------------------------------------------------

    def _emit_batch_edges(
        grad_batched: Tensor | None,
        batch: tuple[list[int], Tensor, Tensor] | None,
        target_node_idxs: list[int],
    ) -> None:
        """Emit edges from one source group to a *batch* of targets.

        ``grad_batched`` is ``(B_targets, [1,] seq, d_model)``; ``batch`` is the
        source ``(node_idx, positions, contrib_mat)``.  Computes the
        ``(B_targets, n_sources)`` weight matrix in one fused op and appends the
        edges above ``min_edge_weight``.
        """
        if grad_batched is None or batch is None:
            return
        src_node_idx, positions, contrib_mat = batch
        g = grad_batched[:, 0] if grad_batched.dim() == 4 else grad_batched  # (B, seq, d_model)
        w = _batched_edge_weights_multi(g, positions, contrib_mat)  # (B_targets, n_sources)
        mask = w.abs() > min_edge_weight
        if not bool(mask.any()):
            return
        bs = mask.nonzero(as_tuple=False).tolist()  # [[target_row, source_col], ...]
        vals = w[mask].tolist()
        edges = graph.edges
        for (tb, sc), weight in zip(bs, vals, strict=True):
            edges.append(
                AttributionEdge(source=src_node_idx[sc], target=target_node_idxs[tb], weight=weight)
            )

    def _edges_for_target_batch(
        target_vec: Tensor,  # (B,) target scalars sharing *source_layers*
        target_node_idxs: list[int],
        source_layers: list[int],
    ) -> None:
        grad_inputs = [embedding] + [residuals[sl] for sl in source_layers]
        b = target_vec.shape[0]
        try:
            cot = torch.eye(b, device=target_vec.device, dtype=target_vec.dtype)
            jacs = torch.autograd.grad(
                target_vec,
                grad_inputs,
                grad_outputs=cot,
                is_grads_batched=True,
                retain_graph=True,
                allow_unused=True,
            )
        except RuntimeError:
            # vmap fallback: per-target backward, then stack (correct, slower).
            cols = [
                torch.autograd.grad(
                    target_vec[j], grad_inputs, retain_graph=True, allow_unused=True
                )
                for j in range(b)
            ]
            jacs = [
                None
                if any(c[ii] is None for c in cols)
                else torch.stack([c[ii] for c in cols], dim=0)
                for ii in range(len(grad_inputs))
            ]
        _emit_batch_edges(jacs[0], embed_batch, target_node_idxs)
        for k, layer in enumerate(source_layers):
            _emit_batch_edges(jacs[k + 1], layer_batches.get(layer), target_node_idxs)

    # --- Feature targets (grouped by layer, chunked) ---
    feature_targets = [(idx, n) for idx, n in enumerate(graph.nodes) if n.node_type == "feature"]
    if max_feature_targets is not None and len(feature_targets) > max_feature_targets:
        # Top-by-activation (selection bias vs circuit-tracer's influence ranking; raise
        # max_feature_targets toward the feature-node count to approach the full matrix).
        feature_targets.sort(key=lambda t: abs(t[1].activation), reverse=True)
        feature_targets = feature_targets[:max_feature_targets]
    n_feat = len(feature_targets)
    log.info(
        "Computing edges for %d feature targets (batched, chunk=%d) ...", n_feat, edge_batch_size
    )

    by_layer_targets: dict[int, list[tuple[int, AttributionNode]]] = {}
    for idx, node in feature_targets:
        by_layer_targets.setdefault(node.layer, []).append((idx, node))

    done = 0
    for layer in sorted(by_layer_targets):
        source_layers = list(range(layer))  # sources are earlier layers (+ embedding)
        pre_act = pre_activations[layer]
        pre2d = pre_act[0] if pre_act.dim() == 3 else pre_act  # (seq, d_transcoder)
        targets = by_layer_targets[layer]
        for s in range(0, len(targets), edge_batch_size):
            chunk = targets[s : s + edge_batch_size]
            pos_idx = torch.tensor([n.position for _, n in chunk], device=pre2d.device)
            feat_idx = torch.tensor([n.feature_idx for _, n in chunk], device=pre2d.device)
            target_vec = pre2d[pos_idx, feat_idx]  # (B,)
            _edges_for_target_batch(target_vec, [i for i, _ in chunk], source_layers)
            done += len(chunk)
        log.info(
            "  layer %d: %d/%d feature targets done (edges: %d)",
            layer,
            done,
            n_feat,
            len(graph.edges),
        )

    # --- Logit targets (batched) ---
    logit_targets = [(idx, n) for idx, n in enumerate(graph.nodes) if n.node_type == "logit"]
    if logit_targets:
        log.info("Computing edges for %d logit targets ...", len(logit_targets))
        pos_idx = torch.tensor([n.position for _, n in logit_targets], device=logits.device)
        tok_idx = torch.tensor([n.token_id for _, n in logit_targets], device=logits.device)
        target_vec = logits[pos_idx, tok_idx]  # (B,)
        _edges_for_target_batch(target_vec, [i for i, _ in logit_targets], list(range(n_layers)))

    return graph
