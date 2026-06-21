"""Tests for llm_circuits.circuits.graph_pruning (no model / GPU / network)."""

from __future__ import annotations

from llm_circuits.circuits.attribution_graph import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
)
from llm_circuits.circuits.graph_pruning import (
    cap_features_by_influence,
    graph_from_dict,
    graph_to_dict,
    prune_graph,
)


def _chain_graph() -> AttributionGraph:
    """emb -> feat(L0) -> feat(L1) -> logit, plus a dangling error and a weak feature."""
    nodes = [
        AttributionNode("embedding", layer=-1, position=0, activation=1.0),  # 0
        AttributionNode("feature", layer=0, position=0, feature_idx=0, activation=1.0),  # 1
        AttributionNode("feature", layer=1, position=0, feature_idx=0, activation=1.0),  # 2
        AttributionNode("logit", layer=2, position=0, token_id=5, activation=10.0),  # 3
        AttributionNode("error", layer=0, position=0, activation=0.5),  # 4 (no outgoing)
        AttributionNode("feature", layer=0, position=0, feature_idx=1, activation=0.01),  # 5 weak
    ]
    edges = [
        AttributionEdge(0, 1, 1.0),  # emb -> feat1
        AttributionEdge(1, 2, 1.0),  # feat1 -> feat2
        AttributionEdge(2, 3, 1.0),  # feat2 -> logit
        AttributionEdge(0, 4, 1.0),  # emb -> error4 (error has no outgoing edge)
        AttributionEdge(0, 5, 0.001),  # emb -> weak feat5
        AttributionEdge(5, 2, 0.001),  # weak feat5 -> feat2
    ]
    return AttributionGraph(nodes=nodes, edges=edges)


class TestSerialization:
    def test_round_trip_preserves_nodes_and_edges(self):
        g = _chain_graph()
        d = graph_to_dict(g)
        g2 = graph_from_dict(d)
        assert len(g2.nodes) == len(g.nodes)
        assert len(g2.edges) == len(g.edges)
        for a, b in zip(g.nodes, g2.nodes, strict=True):
            assert (a.node_type, a.layer, a.position, a.feature_idx, a.token_id) == (
                b.node_type,
                b.layer,
                b.position,
                b.feature_idx,
                b.token_id,
            )
        for a, b in zip(g.edges, g2.edges, strict=True):
            assert (a.source, a.target, a.weight) == (b.source, b.target, b.weight)

    def test_metadata_and_influence_attached(self):
        g = _chain_graph()
        d = graph_to_dict(g, influence_scores=[float(i) for i in range(len(g.nodes))], prompt="hi")
        assert d["prompt"] == "hi"
        assert d["nodes"][2]["influence"] == 2.0


class TestPruneGraph:
    def test_empty_graph(self):
        result = prune_graph(AttributionGraph())
        assert result.graph.nodes == []
        assert result.original_node_count == 0

    def test_keeps_main_path_and_endpoints(self):
        result = prune_graph(_chain_graph(), node_threshold=0.8, edge_threshold=0.95)
        kept = result.graph.nodes
        types = {(n.node_type, n.layer, n.feature_idx) for n in kept}
        # endpoints always kept
        assert any(n.node_type == "embedding" for n in kept)
        assert any(n.node_type == "logit" for n in kept)
        # the high-influence path features survive
        assert ("feature", 0, 0) in types
        assert ("feature", 1, 0) in types

    def test_dangling_error_removed_by_connectivity(self):
        # An error node with no outgoing edge must be pruned regardless of threshold.
        result = prune_graph(_chain_graph(), node_threshold=0.0, edge_threshold=0.0)
        assert all(n.node_type != "error" for n in result.graph.nodes)

    def test_influence_scores_align_with_nodes(self):
        result = prune_graph(_chain_graph())
        assert len(result.influence_scores) == len(result.graph.nodes)

    def test_original_counts_recorded(self):
        g = _chain_graph()
        result = prune_graph(g)
        assert result.original_node_count == len(g.nodes)
        assert result.original_edge_count == len(g.edges)

    def test_reindexed_edges_are_valid(self):
        result = prune_graph(_chain_graph())
        n = len(result.graph.nodes)
        for e in result.graph.edges:
            assert 0 <= e.source < n
            assert 0 <= e.target < n


class TestCapFeaturesByInfluence:
    """circuit-tracer's selection criterion: keep the most *influential* features."""

    def _star_graph(self) -> AttributionGraph:
        # three features feed the logit with decreasing weight -> decreasing influence
        nodes = [
            AttributionNode("embedding", layer=-1, position=0, activation=1.0),  # 0
            AttributionNode("feature", layer=0, position=0, feature_idx=10, activation=0.1),  # 1
            AttributionNode("feature", layer=0, position=0, feature_idx=20, activation=0.1),  # 2
            AttributionNode("feature", layer=0, position=0, feature_idx=30, activation=0.1),  # 3
            AttributionNode("logit", layer=1, position=0, token_id=5, activation=10.0),  # 4
        ]
        edges = [
            AttributionEdge(0, 1, 1.0),
            AttributionEdge(0, 2, 1.0),
            AttributionEdge(0, 3, 1.0),
            AttributionEdge(1, 4, 3.0),  # most influential
            AttributionEdge(2, 4, 2.0),
            AttributionEdge(3, 4, 1.0),  # least influential
        ]
        return AttributionGraph(nodes=nodes, edges=edges)

    def test_keeps_top_features_by_influence_not_activation(self):
        # all three features share activation 0.1, so an activation cap could not rank them;
        # influence ranks by edge weight to the logit -> drop feature_idx=30 first.
        capped = cap_features_by_influence(self._star_graph(), 2)
        feats = {n.feature_idx for n in capped.nodes if n.node_type == "feature"}
        assert feats == {10, 20}  # the two highest-influence features
        # endpoints survive
        assert any(n.node_type == "embedding" for n in capped.nodes)
        assert any(n.node_type == "logit" for n in capped.nodes)

    def test_edges_reindexed_and_dropped_feature_edges_removed(self):
        capped = cap_features_by_influence(self._star_graph(), 2)
        n = len(capped.nodes)
        for e in capped.edges:
            assert 0 <= e.source < n and 0 <= e.target < n
        # 4 nodes (emb + 2 feats + logit), 4 edges (2 in, 2 out); feat-30 edges gone
        assert n == 4
        assert len(capped.edges) == 4

    def test_no_cap_when_none_or_under_limit(self):
        g = self._star_graph()
        assert cap_features_by_influence(g, None) is g
        assert cap_features_by_influence(g, 5) is g  # only 3 features < 5


class TestInfluenceFaithfulToCircuitTracer:
    """Lock node-influence to circuit-tracer's reference implementation."""

    def test_matches_circuit_tracer(self):
        import numpy as np
        import pytest

        ct = pytest.importorskip("circuit_tracer.graph")
        import torch

        from llm_circuits.circuits.graph_pruning import (
            _build_adjacency_matrix,
            _compute_influence,
            _normalize_matrix,
        )

        # nodes [features, errors, tokens, logits] (circuit-tracer order); edges (src, tgt, w)
        e = [(3, 0, 1.0), (0, 1, 2.0), (1, 4, 3.0), (2, 4, 0.5), (3, 4, 0.1)]
        n = 5
        nodes = [
            AttributionNode("feature", layer=0, position=0),
            AttributionNode("feature", layer=1, position=0),
            AttributionNode("error", layer=0, position=0),
            AttributionNode("embedding", layer=-1, position=0),
            AttributionNode("logit", layer=2, position=0, token_id=9),
        ]
        edges = [AttributionEdge(source=s, target=t, weight=w) for s, t, w in e]
        lw = np.zeros(n)
        lw[4] = 1.0

        ours = _compute_influence(_normalize_matrix(_build_adjacency_matrix(nodes, edges)), lw)

        a_ct = torch.zeros(n, n, dtype=torch.float64)
        for s, t, w in e:
            a_ct[t, s] += w
        theirs = ct.compute_node_influence(a_ct, torch.from_numpy(lw)).numpy()

        assert np.abs(ours - theirs).max() < 1e-6
