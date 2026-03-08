"""Graph pruning for attribution graphs.

Removes low-influence nodes and edges from an :class:`AttributionGraph`,
producing a compact subgraph suitable for human inspection.  The algorithm
is adapted from circuit-tracer's ``prune_graph`` but operates on our
list-based graph representation using **numpy** (no GPU required).

Public API
----------
- :func:`prune_graph` — prune an :class:`AttributionGraph` in-memory.
- :func:`prune_graph_dict` — prune from/to a JSON-compatible dict.
- :func:`graph_from_dict` / :func:`graph_to_dict` — JSON round-trip helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_circuits.circuits.attribution_graph import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PrunedGraph:
    """Result of :func:`prune_graph`."""

    graph: AttributionGraph
    influence_scores: list[float]
    """Per-node influence score (aligned with ``graph.nodes``)."""
    original_node_count: int
    original_edge_count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_adjacency_matrix(
    nodes: list[AttributionNode],
    edges: list[AttributionEdge],
) -> np.ndarray:
    """Build a dense ``(N, N)`` adjacency matrix from *edges*.

    ``A[source, target] = weight``.  Multiple edges between the same pair
    are summed.
    """
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        A[e.source, e.target] += e.weight
    return A


def _compute_logit_weights(nodes: list[AttributionNode]) -> np.ndarray:
    """Softmax over logit-node activations → probability vector of length *N*."""
    n = len(nodes)
    w = np.zeros(n, dtype=np.float64)
    logit_indices = [i for i, nd in enumerate(nodes) if nd.node_type == "logit"]
    if not logit_indices:
        return w
    logit_vals = np.array([nodes[i].activation for i in logit_indices])
    # Numerically-stable softmax
    logit_vals = logit_vals - logit_vals.max()
    exp_vals = np.exp(logit_vals)
    probs = exp_vals / exp_vals.sum()
    for idx, li in enumerate(logit_indices):
        w[li] = probs[idx]
    return w


def _normalize_matrix(A: np.ndarray) -> np.ndarray:
    """Row-normalize ``|A|`` (each row sums to 1, or 0 if the row is zero)."""
    absA = np.abs(A)
    row_sums = absA.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    return absA / row_sums


def _compute_influence(
    A_norm: np.ndarray,
    logit_weights: np.ndarray,
    *,
    max_iter: int = 1000,
) -> np.ndarray:
    """Compute per-node influence via power iteration.

    Iterates ``influence = w @ A + w @ A^2 + ...`` until convergence.
    Uses the *transposed* normalised adjacency so that influence flows
    backward from logits to upstream nodes.
    """
    At = A_norm.T  # influence propagates backward
    current = logit_weights @ At
    influence = current.copy()
    for _ in range(max_iter):
        if not np.any(current > 0):
            break
        current = current @ At
        influence += current
    return influence


def _find_threshold(scores: np.ndarray, threshold: float) -> float:
    """Return the minimum score such that keeping all scores >= it accounts
    for at least *threshold* of the total score mass.
    """
    sorted_scores = np.sort(scores)[::-1]  # descending
    total = sorted_scores.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(sorted_scores) / total
    idx = int(np.searchsorted(cum, threshold))
    idx = min(idx, len(cum) - 1)
    return float(sorted_scores[idx])


def _connectivity_prune(
    nodes: list[AttributionNode],
    kept_nodes: np.ndarray,
    kept_edges: np.ndarray,
    edges: list[AttributionEdge],
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively remove feature/error nodes lacking incoming or outgoing edges."""
    n_nodes = len(nodes)

    # Determine which nodes are "internal" (feature or error)
    is_feature = np.array([nd.node_type == "feature" for nd in nodes])
    is_internal = np.array([nd.node_type in ("feature", "error") for nd in nodes])

    changed = True
    while changed:
        changed = False

        # Build incoming/outgoing masks from kept edges
        has_incoming = np.zeros(n_nodes, dtype=bool)
        has_outgoing = np.zeros(n_nodes, dtype=bool)
        for i, e in enumerate(edges):
            if kept_edges[i]:
                has_outgoing[e.source] = True
                has_incoming[e.target] = True

        # Internal nodes (feature + error) must have outgoing edges
        remove = is_internal & kept_nodes & ~has_outgoing
        if remove.any():
            kept_nodes[remove] = False
            changed = True

        # Feature nodes must also have incoming edges
        remove = is_feature & kept_nodes & ~has_incoming
        if remove.any():
            kept_nodes[remove] = False
            changed = True

        # Remove edges touching removed nodes
        for i, e in enumerate(edges):
            if kept_edges[i] and (not kept_nodes[e.source] or not kept_nodes[e.target]):
                kept_edges[i] = False
                changed = True

    return kept_nodes, kept_edges


def _reindex_graph(
    graph: AttributionGraph,
    kept_node_mask: np.ndarray,
    kept_edge_mask: np.ndarray,
    influence: np.ndarray,
) -> PrunedGraph:
    """Build a new :class:`AttributionGraph` with remapped indices."""
    old_to_new: dict[int, int] = {}
    new_nodes: list[AttributionNode] = []
    new_influence: list[float] = []
    for old_idx, node in enumerate(graph.nodes):
        if kept_node_mask[old_idx]:
            new_idx = len(new_nodes)
            old_to_new[old_idx] = new_idx
            new_nodes.append(node)
            new_influence.append(float(influence[old_idx]))

    new_edges: list[AttributionEdge] = []
    for i, edge in enumerate(graph.edges):
        if kept_edge_mask[i]:
            new_edges.append(
                AttributionEdge(
                    source=old_to_new[edge.source],
                    target=old_to_new[edge.target],
                    weight=edge.weight,
                )
            )

    return PrunedGraph(
        graph=AttributionGraph(nodes=new_nodes, edges=new_edges),
        influence_scores=new_influence,
        original_node_count=len(graph.nodes),
        original_edge_count=len(graph.edges),
    )


# ---------------------------------------------------------------------------
# Public API — in-memory
# ---------------------------------------------------------------------------


def prune_graph(
    graph: AttributionGraph,
    *,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> PrunedGraph:
    """Prune *graph* by removing low-influence nodes and edges.

    Parameters
    ----------
    graph:
        The attribution graph to prune.
    node_threshold:
        Keep nodes contributing to this fraction of total influence (0-1).
    edge_threshold:
        Keep edges contributing to this fraction of total influence (0-1).

    Returns
    -------
    PrunedGraph
        The pruned graph with influence scores.
    """
    if not (0.0 <= node_threshold <= 1.0):
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if not (0.0 <= edge_threshold <= 1.0):
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    nodes = graph.nodes
    edges = graph.edges
    n = len(nodes)

    if n == 0:
        return PrunedGraph(
            graph=AttributionGraph(),
            influence_scores=[],
            original_node_count=0,
            original_edge_count=len(edges),
        )

    # Step 1: build adjacency and compute influence
    A = _build_adjacency_matrix(nodes, edges)
    logit_weights = _compute_logit_weights(nodes)
    A_norm = _normalize_matrix(A)
    influence = _compute_influence(A_norm, logit_weights)

    # Step 2: node threshold
    node_cutoff = _find_threshold(influence, node_threshold)
    kept_nodes = influence >= node_cutoff

    # Always keep embedding and logit nodes
    for i, nd in enumerate(nodes):
        if nd.node_type in ("embedding", "logit"):
            kept_nodes[i] = True

    # Step 3: zero out pruned nodes in the adjacency matrix for edge scoring
    pruned_A = A.copy()
    pruned_A[~kept_nodes, :] = 0.0
    pruned_A[:, ~kept_nodes] = 0.0

    # Step 4: edge threshold — compute per-edge influence scores
    A_norm_pruned = _normalize_matrix(pruned_A)
    At = A_norm_pruned.T
    # Forward pass of logit weights through the normalised transpose
    edge_scores_matrix = np.zeros_like(pruned_A)
    current = logit_weights.copy()
    for _ in range(1000):
        contrib = current[:, None] * At
        edge_scores_matrix += contrib
        current = current @ At
        if not np.any(current > 0):
            break

    # Flatten edge scores to per-edge list
    kept_edges = np.zeros(len(edges), dtype=bool)
    edge_score_vals = np.zeros(len(edges), dtype=np.float64)
    for i, e in enumerate(edges):
        edge_score_vals[i] = edge_scores_matrix[e.target, e.source]

    edge_cutoff = _find_threshold(edge_score_vals, edge_threshold)
    kept_edges = edge_score_vals >= edge_cutoff

    # Also discard edges touching pruned nodes
    for i, e in enumerate(edges):
        if not kept_nodes[e.source] or not kept_nodes[e.target]:
            kept_edges[i] = False

    # Step 5: connectivity pruning
    kept_nodes, kept_edges = _connectivity_prune(nodes, kept_nodes, kept_edges, edges)

    # Step 6: reindex
    return _reindex_graph(graph, kept_nodes, kept_edges, influence)


# ---------------------------------------------------------------------------
# Public API — JSON round-trip
# ---------------------------------------------------------------------------


def graph_from_dict(d: dict) -> AttributionGraph:
    """Deserialize an :class:`AttributionGraph` from a JSON-compatible dict."""
    nodes = [
        AttributionNode(
            node_type=nd["node_type"],
            layer=nd["layer"],
            position=nd["position"],
            feature_idx=nd.get("feature_idx"),
            token_id=nd.get("token_id"),
            activation=nd.get("activation", 0.0),
            label=nd.get("label"),
        )
        for nd in d.get("nodes", [])
    ]
    edges = [
        AttributionEdge(
            source=ed["source"],
            target=ed["target"],
            weight=ed["weight"],
        )
        for ed in d.get("edges", [])
    ]
    return AttributionGraph(nodes=nodes, edges=edges)


def graph_to_dict(
    graph: AttributionGraph,
    *,
    influence_scores: list[float] | None = None,
    **metadata: object,
) -> dict:
    """Serialize an :class:`AttributionGraph` to a JSON-compatible dict.

    Extra *metadata* keyword arguments are included at the top level.
    """
    d: dict = {}
    d.update(metadata)
    d["nodes"] = [
        {
            "node_type": nd.node_type,
            "layer": nd.layer,
            "position": nd.position,
            "feature_idx": nd.feature_idx,
            "token_id": nd.token_id,
            "activation": nd.activation,
            "label": nd.label,
        }
        for nd in graph.nodes
    ]
    if influence_scores is not None:
        for i, score in enumerate(influence_scores):
            d["nodes"][i]["influence"] = score
    d["edges"] = [
        {
            "source": e.source,
            "target": e.target,
            "weight": e.weight,
        }
        for e in graph.edges
    ]
    return d


def prune_graph_dict(
    graph_dict: dict,
    *,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> dict:
    """Prune an attribution graph from its JSON dict representation.

    Preserves all top-level metadata keys (e.g. ``prompt``, ``model``).
    """
    graph = graph_from_dict(graph_dict)
    result = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)

    # Preserve metadata
    metadata = {k: v for k, v in graph_dict.items() if k not in ("nodes", "edges")}
    metadata["pruning"] = {
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
        "original_node_count": result.original_node_count,
        "original_edge_count": result.original_edge_count,
    }
    return graph_to_dict(result.graph, influence_scores=result.influence_scores, **metadata)
