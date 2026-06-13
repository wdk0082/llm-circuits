"""Tests for llm_circuits.circuits.graph_pruning (no model / GPU / network)."""

from __future__ import annotations

from llm_circuits.circuits.attribution_graph import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
)
from llm_circuits.circuits.graph_pruning import (
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
