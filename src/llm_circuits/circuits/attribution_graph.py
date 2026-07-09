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

import numpy as np
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

    prob: float | None = None
    """Softmax probability over the full vocab (only for ``"logit"`` nodes).

    circuit-tracer seeds influence with the *actual* probabilities of the selected
    logits (which sum to ``desired_logit_prob``), not a renormalised softmax over only
    the selected ones — so we store it here rather than re-deriving from ``activation``.
    """

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

    Sources are grouped by position so each group is a single ``(B, d) @ (d, n_p)``
    matmul — this avoids materialising the ``(B, n_src, d_model)`` intermediate, which
    would OOM when ``n_src`` is large (e.g. influence mode, where *every* feature is a
    potential source).  ``seq`` is small, so the per-position loop is cheap.
    """
    out = contrib_mat.new_zeros(grad_3d.shape[0], positions.shape[0])  # (B, n_src)
    for p in torch.unique(positions):
        mask = positions == p
        gp = grad_3d[:, int(p), :].to(contrib_mat.dtype)  # (B, d_model)
        out[:, mask] = gp @ contrib_mat[mask].T  # (B, n_p)
    return out


def _select_salient_logits(
    logit_vec: Tensor, desired_logit_prob: float, max_n_logits: int
) -> tuple[Tensor, Tensor]:
    """Pick the smallest logit set whose cumulative softmax prob >= desired_logit_prob.

    Mirrors circuit-tracer's ``compute_salient_logits`` index/probability logic exactly
    (top-k by prob, then a cumulative-probability cutoff capped at ``max_n_logits``).
    Returns ``(token_ids, probs)`` for the selected logits (probs sum to ~desired_prob).
    """
    probs = torch.softmax(logit_vec, dim=-1)
    top_p, top_idx = probs.topk(max_n_logits)
    cutoff = int(torch.searchsorted(top_p.cumsum(0), desired_logit_prob)) + 1
    cutoff = min(cutoff, max_n_logits)
    return top_idx[:cutoff], top_p[:cutoff]


def _partial_feature_influence(
    n_nodes: int,
    tgt: np.ndarray,
    src: np.ndarray,
    weight: np.ndarray,
    logit_nodes: np.ndarray,
    logit_p: np.ndarray,
    *,
    max_iter: int = 128,
) -> np.ndarray:
    """Power-iteration node influence over the *partially-attributed* graph.

    Mirrors circuit-tracer's ``compute_partial_influences`` but sparse (``bincount``
    scatter-adds over the edge arrays) so it scales to ~10^5 nodes without a dense
    ``N*N`` matrix.  Only attributed *targets* (nodes that already have incoming edges)
    propagate influence back to their sources, so an as-yet-unattributed feature accrues
    influence purely from the targets it feeds — exactly the signal used to decide which
    feature to attribute next.

    ``tgt``/``src``/``weight`` are parallel arrays over the currently-attributed edges
    (``weight`` indexed ``[target, source]``); ``logit_nodes``/``logit_p`` seed the
    logits with their probabilities.  Returns an influence score per node index.
    """
    influence = np.zeros(n_nodes, dtype=np.float64)
    if tgt.size == 0:
        return influence
    absw = np.abs(weight)
    # Row-normalise over each target's incoming edges (sum of |w| per target).
    row_sum = np.bincount(tgt, weights=absw, minlength=n_nodes)
    w_norm = absw / np.maximum(row_sum[tgt], 1e-10)

    prod = np.zeros(n_nodes, dtype=np.float64)
    prod[logit_nodes] = logit_p
    for _ in range(max_iter):
        # distribute each target's current mass to its sources, weighted by w_norm
        prod = np.bincount(src, weights=prod[tgt] * w_norm, minlength=n_nodes)
        if not prod.any():
            break
        influence += prod
    return influence


def _select_features_by_influence(
    n_nodes: int,
    feat_node_arr: np.ndarray,
    logit_nodes: np.ndarray,
    logit_p: np.ndarray,
    target_cap: int,
    get_edges: Any,
    attribute: Any,
    *,
    batch_size: int,
    update_interval: int,
) -> np.ndarray:
    """circuit-tracer's dynamic feature selection (``_run_attribution``'s Phase 4 loop).

    Repeatedly: rank features by influence over the edges attributed *so far*
    (``get_edges()`` -> ``(tgt, src, weight)`` arrays), then ``attribute(chunk)`` the most
    influential unvisited ones in batches, re-ranking every ``update_interval`` batches,
    until ``target_cap`` features have been attributed.  Pure orchestration (no torch) so
    it can be unit-tested against a synthetic edge oracle.  Returns a boolean ``visited``
    mask over node indices.
    """
    visited = np.zeros(n_nodes, dtype=bool)
    n_visited = 0
    while n_visited < target_cap:
        tgt, src, wgt = get_edges()
        infl = _partial_feature_influence(n_nodes, tgt, src, wgt, logit_nodes, logit_p)
        order = feat_node_arr[np.argsort(-infl[feat_node_arr])]
        pending = [int(i) for i in order if not visited[i]][: update_interval * batch_size]
        if not pending:
            break  # no more reachable/influential features
        for s in range(0, len(pending), batch_size):
            if n_visited >= target_cap:
                break
            chunk = pending[s : s + batch_size]
            if n_visited + len(chunk) > target_cap:
                chunk = chunk[: target_cap - n_visited]
            attribute(chunk)
            visited[chunk] = True
            n_visited += len(chunk)
    return visited


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_attribution_graph(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    *,
    n_bos_tokens: int = 1,
    desired_logit_prob: float = 0.95,
    max_n_logits: int = 10,
    max_feature_targets: int | None = None,
    max_feature_nodes: int | None = None,
    max_targets_guard: int | None = None,
    feature_selection: str = "all",
    update_interval: int = 4,
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
    if not is_set:
        # Cross-layer transcoders would need every source feature's decoder contribution
        # summed over ALL its output layers (a_s * sum_l W_dec^{l_s->l} . grad_l), not just
        # the self-layer term — the paper's edge formula.  The Phase-4 code below only
        # takes offset 0, which silently under-counts CLT edges, and with the registry now
        # Qwen3-only (per-layer) there is no CLT model left to validate a fix against.
        raise NotImplementedError(
            "build_attribution_graph supports per-layer transcoders (TranscoderSet) only; "
            "cross-layer transcoder edges (sum over output layers) are not implemented."
        )
    n_layers = len(transcoder)

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
    # In "influence_ranked" mode every active feature is kept as a NODE (a potential
    # source/column); max_feature_nodes instead caps how many become attributed TARGETS
    # (chosen by influence in Phase 5). The activation pre-cap applies only to "all" mode.
    if (
        feature_selection != "influence_ranked"
        and max_feature_nodes is not None
        and len(feat_candidates) > max_feature_nodes
    ):
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

    # Logit nodes: circuit-tracer's `compute_salient_logits` — the smallest set whose
    # cumulative softmax probability reaches desired_logit_prob, capped at max_n_logits.
    last_pos = seq_len - 1
    top_idx, top_p = _select_salient_logits(logits[last_pos], desired_logit_prob, max_n_logits)
    cutoff = len(top_idx)
    log.info("Selected %d logits with cumulative probability %.4f", cutoff, top_p.sum().item())
    for rank in range(cutoff):
        tok = int(top_idx[rank])
        idx = len(graph.nodes)
        node_index[("logit", n_layers, last_pos, tok)] = idx
        graph.nodes.append(
            AttributionNode(
                node_type="logit",
                layer=n_layers,
                position=last_pos,
                token_id=tok,
                activation=logits[last_pos, tok].item(),  # raw logit value (display / demean ref)
                prob=float(top_p[rank]),  # actual prob (influence seed; sums to ~desired_prob)
            )
        )

    n_feature_nodes = sum(1 for n in graph.nodes if n.node_type == "feature")
    log.info(
        "Built %d nodes (emb=%d, feat=%d, err=%d, logit=%d)",
        len(graph.nodes),
        sum(1 for n in graph.nodes if n.node_type == "embedding"),
        n_feature_nodes,
        sum(1 for n in graph.nodes if n.node_type == "error"),
        sum(1 for n in graph.nodes if n.node_type == "logit"),
    )

    # Fail FAST — before the expensive contrib-vector (Phase 4) and edge (Phase 5) work —
    # if this build would attribute more feature targets than the guard allows. A dense
    # prompt with no cap (or influence with no cap) otherwise wastes a long Phase 4 over
    # ~10^6 features before tripping. The dense N*N prune matrix scales as planned^2.
    if max_targets_guard is not None:
        if feature_selection == "influence_ranked":
            planned = min(max_feature_nodes or n_feature_nodes, n_feature_nodes)
        else:
            planned = min(n_feature_nodes, max_feature_targets or n_feature_nodes)
        if planned > max_targets_guard:
            raise ValueError(
                f"This build would attribute {planned} feature targets "
                f"(guard={max_targets_guard}); on a dense prompt the dense N*N prune matrix "
                f"would exhaust memory. Set a feat-nodes cap (e.g. 8000) — influence mode "
                f"keeps the most influential features, so a cap is near-lossless — or use a "
                f"shorter prompt for no-cap mode."
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
        # Load W_dec once for this layer (handles lazy loading internally).
        # Per-layer transcoders only (CLT rejected above): the decoder writes to its
        # own layer, so the source contribution is a_s * W_dec at that layer.
        dec_vecs = transcoder.transcoders[layer]._get_decoder_vectors(feat_indices)

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

    # Compute edges for a set of feature targets, grouped by layer + chunked.
    def _attribute_feature_targets(
        targets: list[tuple[int, AttributionNode]], n_feat: int | None = None
    ) -> None:
        by_layer_targets: dict[int, list[tuple[int, AttributionNode]]] = {}
        for idx, node in targets:
            by_layer_targets.setdefault(node.layer, []).append((idx, node))
        done = 0
        for layer in sorted(by_layer_targets):
            source_layers = list(range(layer))  # sources are earlier layers (+ embedding)
            pre_act = pre_activations[layer]
            pre2d = pre_act[0] if pre_act.dim() == 3 else pre_act  # (seq, d_transcoder)
            ts = by_layer_targets[layer]
            for s in range(0, len(ts), edge_batch_size):
                chunk = ts[s : s + edge_batch_size]
                pos_idx = torch.tensor([n.position for _, n in chunk], device=pre2d.device)
                feat_idx = torch.tensor([n.feature_idx for _, n in chunk], device=pre2d.device)
                target_vec = pre2d[pos_idx, feat_idx]  # (B,)
                _edges_for_target_batch(target_vec, [i for i, _ in chunk], source_layers)
                done += len(chunk)
            if n_feat is not None:
                log.info(
                    "  layer %d: %d/%d feature targets done (edges: %d)",
                    layer,
                    done,
                    n_feat,
                    len(graph.edges),
                )

    def _attribute_logits() -> None:
        logit_targets = [(idx, n) for idx, n in enumerate(graph.nodes) if n.node_type == "logit"]
        if not logit_targets:
            return
        log.info("Computing edges for %d logit targets ...", len(logit_targets))
        pos_idx = torch.tensor([n.position for _, n in logit_targets], device=logits.device)
        tok_idx = torch.tensor([n.token_id for _, n in logit_targets], device=logits.device)
        # Demeaned logit direction (circuit-tracer): attribute (logit_t - mean_v logit_v) so
        # the gradient seed is the unembedding column minus the mean unembedding direction.
        # Softmax is invariant to a constant shift, so this is the part that affects p.
        logit_mean = logits.mean(dim=-1)  # (seq,) mean over vocab per position
        target_vec = logits[pos_idx, tok_idx] - logit_mean[pos_idx]  # (B,)
        _edges_for_target_batch(target_vec, [i for i, _ in logit_targets], list(range(n_layers)))

    all_feature_targets = [
        (idx, n) for idx, n in enumerate(graph.nodes) if n.node_type == "feature"
    ]

    # --- Influence-ranked feature attribution (circuit-tracer's dynamic selection) ---
    if feature_selection == "influence_ranked":
        _attribute_logits()  # logits first: they seed the influence propagation
        logit_nodes = np.array(
            [i for i, n in enumerate(graph.nodes) if n.node_type == "logit"], dtype=np.int64
        )
        # Seed with the actual softmax probabilities (circuit-tracer), not a renormalised
        # softmax over the selected logits — these sum to ~desired_logit_prob.
        logit_p = np.array([graph.nodes[i].prob or 0.0 for i in logit_nodes], dtype=np.float64)

        n_total = len(all_feature_targets)
        target_cap = min(max_feature_nodes or n_total, n_total)  # guarded above, pre-Phase-4
        n_nodes = len(graph.nodes)
        feat_node_arr = np.array([idx for idx, _ in all_feature_targets], dtype=np.int64)
        node_by_idx = {idx: node for idx, node in all_feature_targets}
        log.info(
            "Influence-ranked attribution: selecting %d of %d feature nodes ...",
            target_cap,
            n_total,
        )

        def _get_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            m = len(graph.edges)
            return (
                np.fromiter((e.target for e in graph.edges), np.int64, m),
                np.fromiter((e.source for e in graph.edges), np.int64, m),
                np.fromiter((e.weight for e in graph.edges), np.float64, m),
            )

        def _attribute(chunk: list[int]) -> None:
            _attribute_feature_targets([(i, node_by_idx[i]) for i in chunk])
            log.info("  attributed up to feature node (edges: %d)", len(graph.edges))

        visited = _select_features_by_influence(
            n_nodes,
            feat_node_arr,
            logit_nodes,
            logit_p,
            target_cap,
            _get_edges,
            _attribute,
            batch_size=edge_batch_size,
            update_interval=update_interval,
        )

        # Drop features that were never attributed (and edges touching them); reindex.
        keep = [(n.node_type != "feature") or bool(visited[i]) for i, n in enumerate(graph.nodes)]
        old_to_new = {old: i for i, old in enumerate(o for o in range(len(graph.nodes)) if keep[o])}
        new_nodes = [n for i, n in enumerate(graph.nodes) if keep[i]]
        new_edges = [
            AttributionEdge(old_to_new[e.source], old_to_new[e.target], e.weight)
            for e in graph.edges
            if keep[e.source] and keep[e.target]
        ]
        return AttributionGraph(nodes=new_nodes, edges=new_edges)

    # --- "all" mode: attribute every kept feature node (optionally capped by activation) ---
    feature_targets = all_feature_targets
    if max_feature_targets is not None and len(feature_targets) > max_feature_targets:
        # Top-by-activation (selection bias vs circuit-tracer's influence ranking; raise
        # max_feature_targets toward the feature-node count to approach the full matrix).
        feature_targets.sort(key=lambda t: abs(t[1].activation), reverse=True)
        feature_targets = feature_targets[:max_feature_targets]
    n_feat = len(feature_targets)  # guarded above (pre-Phase-4)
    log.info(
        "Computing edges for %d feature targets (batched, chunk=%d) ...", n_feat, edge_batch_size
    )
    _attribute_feature_targets(feature_targets, n_feat=n_feat)
    _attribute_logits()
    return graph
