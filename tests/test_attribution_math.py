"""Tests for the batched edge-weight math in attribution_graph (CPU, no model)."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from llm_circuits.circuits.attribution_graph import (
    _batched_edge_weights,
    _batched_edge_weights_multi,
    _partial_feature_influence,
    _select_features_by_influence,
    _select_salient_logits,
    build_attribution_graph,
)


def _reference_weights(grad_2d, positions, contribs):
    """The old per-source dot-product loop, used as ground truth."""
    out = []
    for i in range(len(contribs)):
        g = grad_2d[positions[i].item(), :]
        out.append((g.float() @ contribs[i].float()).item())
    return out


class TestBatchedEdgeWeights:
    def test_matches_per_source_loop(self):
        torch.manual_seed(0)
        seq, d, n = 10, 48, 30
        grad_2d = torch.randn(seq, d)
        positions = torch.randint(0, seq, (n,))
        contribs = [torch.randn(d) for _ in range(n)]
        contrib_mat = torch.stack(contribs).float()

        got = _batched_edge_weights(grad_2d, positions, contrib_mat).tolist()
        ref = _reference_weights(grad_2d, positions, contribs)
        assert max(abs(a - b) for a, b in zip(got, ref, strict=True)) < 1e-4

    def test_shape_and_single_source(self):
        grad_2d = torch.randn(4, 8)
        positions = torch.tensor([2])
        contrib_mat = torch.randn(1, 8)
        w = _batched_edge_weights(grad_2d, positions, contrib_mat)
        assert w.shape == (1,)
        expected = (grad_2d[2] * contrib_mat[0]).sum()
        assert torch.allclose(w[0], expected, atol=1e-5)

    def test_repeated_positions(self):
        # Several sources reading from the same position is the common case.
        grad_2d = torch.randn(3, 6)
        positions = torch.tensor([1, 1, 1])
        contrib_mat = torch.randn(3, 6)
        w = _batched_edge_weights(grad_2d, positions, contrib_mat)
        for i in range(3):
            assert torch.allclose(w[i], (grad_2d[1] * contrib_mat[i]).sum(), atol=1e-5)

    def test_casts_to_contrib_dtype(self):
        grad_2d = torch.randn(4, 5, dtype=torch.float64)
        positions = torch.tensor([0, 3])
        contrib_mat = torch.randn(2, 5, dtype=torch.float32)
        w = _batched_edge_weights(grad_2d, positions, contrib_mat)
        assert w.dtype == torch.float32


class TestBatchedEdgeWeightsMulti:
    """The multi-target helper must equal the single-target one, per target row."""

    def test_matches_single_target_per_row(self):
        torch.manual_seed(1)
        b, seq, d, n = 7, 12, 32, 20
        grad_3d = torch.randn(b, seq, d)
        positions = torch.randint(0, seq, (n,))
        contrib_mat = torch.randn(n, d)
        multi = _batched_edge_weights_multi(grad_3d, positions, contrib_mat)
        assert multi.shape == (b, n)
        for t in range(b):
            single = _batched_edge_weights(grad_3d[t], positions, contrib_mat)
            assert torch.allclose(multi[t], single, atol=1e-6)

    def test_casts_to_contrib_dtype(self):
        grad_3d = torch.randn(3, 4, 5, dtype=torch.float64)
        positions = torch.tensor([0, 3])
        contrib_mat = torch.randn(2, 5, dtype=torch.float32)
        assert _batched_edge_weights_multi(grad_3d, positions, contrib_mat).dtype == torch.float32


def test_build_attribution_graph_exposes_knobs():
    params = inspect.signature(build_attribution_graph).parameters
    assert "min_edge_weight" in params
    assert params["min_edge_weight"].default == 0.0
    assert "edge_batch_size" in params  # batched-backward chunk size
    assert "feature_selection" in params  # "all" | "influence_ranked"
    assert "update_interval" in params  # influence re-ranking cadence
    # circuit-tracer logit selection (replaces the old fixed top_k_logits)
    assert params["desired_logit_prob"].default == 0.95
    assert params["max_n_logits"].default == 10


class TestSalientLogits:
    """Logit selection + demeaning must match circuit-tracer's compute_salient_logits."""

    def test_selection_matches_circuit_tracer(self):
        import pytest

        cs = pytest.importorskip("circuit_tracer.utils.salient_logits")
        torch.manual_seed(3)
        d_vocab, d_model = 200, 16
        logits = torch.randn(d_vocab) * 2.0
        w_u = torch.randn(d_model, d_vocab)  # (d_model, d_vocab)

        idx_ct, p_ct, _ = cs.compute_salient_logits(
            logits, w_u, max_n_logits=10, desired_logit_prob=0.95
        )
        idx_ours, p_ours = _select_salient_logits(logits, desired_logit_prob=0.95, max_n_logits=10)

        assert idx_ours.tolist() == idx_ct.tolist()  # same token set, same order
        assert torch.allclose(p_ours, p_ct, atol=1e-6)  # same actual probabilities

    def test_demeaned_gradient_matches_unembed_demean(self):
        # Our logit target (logit_t - mean_v logit_v) has gradient W_U[:,t] - mean_v W_U[:,v],
        # which is exactly circuit-tracer's demeaned unembedding direction.
        import pytest

        cs = pytest.importorskip("circuit_tracer.utils.salient_logits")
        torch.manual_seed(4)
        d_vocab, d_model = 50, 8
        resid = torch.randn(d_model)
        w_u = torch.randn(d_model, d_vocab)
        logits = resid @ w_u  # (d_vocab,)

        _, _, demeaned_vecs = cs.compute_salient_logits(
            logits, w_u, max_n_logits=5, desired_logit_prob=0.95
        )
        idx, _ = _select_salient_logits(logits, 0.95, 5)
        # gradient of (logit_t - mean_v logit_v) wrt resid = W_U[:,t] - W_U.mean(dim=1)
        ours_dirs = (w_u[:, idx] - w_u.mean(dim=1, keepdim=True)).T  # (k, d_model)
        assert torch.allclose(ours_dirs, demeaned_vecs, atol=1e-5)


# A tiny partially-attributed graph used by the influence tests below.
# Node order: 0..3 = features, 4 = error/source-only, 5 = logit (LAST, as circuit-tracer
# requires). Attributed targets so far: logit(5) and features 0,1. Edge = (target, source, w).
_EDGES = [
    (5, 0, 2.0),
    (5, 1, 1.0),
    (5, 2, 0.5),
    (0, 2, 3.0),
    (0, 4, 1.0),
    (1, 3, 4.0),
]
_N_NODES = 6
_LOGIT_NODES = np.array([5])
_LOGIT_P = np.array([1.0])


def _edge_arrays(edges):
    tgt = np.array([t for t, _, _ in edges], dtype=np.int64)
    src = np.array([s for _, s, _ in edges], dtype=np.int64)
    w = np.array([w for _, _, w in edges], dtype=np.float64)
    return tgt, src, w


class TestPartialInfluence:
    def test_matches_circuit_tracer_compute_partial_influences(self):
        """Lock _partial_feature_influence to circuit-tracer's reference, exactly."""
        import pytest

        ct = pytest.importorskip("circuit_tracer.graph")

        tgt, src, w = _edge_arrays(_EDGES)
        ours = _partial_feature_influence(_N_NODES, tgt, src, w, _LOGIT_NODES, _LOGIT_P)

        # Build circuit-tracer's rectangular edge_matrix: one row per attributed target,
        # columns over all nodes; row_to_node_index maps rows -> node ids (logits last).
        rows = [5, 0, 1]  # attributed targets in our edge set
        row_to_node_index = torch.tensor(rows, dtype=torch.long)
        # circuit-tracer's internal `prod` is float32, so feed float32 to avoid a dtype clash.
        edge_matrix = torch.zeros(len(rows), _N_NODES, dtype=torch.float32)
        for t, s, weight in _EDGES:
            edge_matrix[rows.index(t), s] = weight
        theirs = ct.compute_partial_influences(
            edge_matrix, torch.tensor([1.0], dtype=torch.float32), row_to_node_index, device="cpu"
        ).numpy()

        assert np.abs(ours - theirs).max() < 1e-6

    def test_fully_attributed_equals_dense_full_influence(self):
        """When every target is attributed, partial influence == the dense full influence."""
        from llm_circuits.circuits.attribution_graph import AttributionEdge, AttributionNode
        from llm_circuits.circuits.graph_pruning import (
            _build_adjacency_matrix,
            _compute_influence,
            _normalize_matrix,
        )

        nodes = [
            AttributionNode("feature", layer=0, position=0),  # 0
            AttributionNode("feature", layer=1, position=0),  # 1
            AttributionNode("feature", layer=0, position=0),  # 2
            AttributionNode("feature", layer=1, position=0),  # 3
            AttributionNode("error", layer=0, position=0),  # 4
            AttributionNode("logit", layer=2, position=0, token_id=9, activation=0.0),  # 5
        ]
        edges = [AttributionEdge(source=s, target=t, weight=w) for t, s, w in _EDGES]
        dense = _compute_influence(
            _normalize_matrix(_build_adjacency_matrix(nodes, edges)),
            np.array([0, 0, 0, 0, 0, 1.0]),  # logit weight on node 5
        )
        tgt, src, w = _edge_arrays(_EDGES)
        ours = _partial_feature_influence(_N_NODES, tgt, src, w, _LOGIT_NODES, _LOGIT_P)
        assert np.allclose(ours, dense, atol=1e-9)


class TestInfluenceRankedSelection:
    """The dynamic selection loop must keep the most influential features."""

    def test_single_layer_picks_top_k_by_influence(self):
        # features 0..4 feed the logit (5) with decreasing weight -> decreasing influence.
        logit_edges = [(5, 0, 5.0), (5, 1, 4.0), (5, 2, 3.0), (5, 3, 2.0), (5, 4, 1.0)]
        store = list(logit_edges)  # logits attributed first (as in build)
        oracle = {0: [], 1: [], 2: [], 3: [], 4: []}  # no incoming edges (2-layer graph)

        def get_edges():
            return _edge_arrays(store)

        def attribute(chunk):
            for f in chunk:
                store.extend(oracle[f])

        visited = _select_features_by_influence(
            6,
            np.array([0, 1, 2, 3, 4]),
            np.array([5]),
            np.array([1.0]),
            target_cap=3,
            get_edges=get_edges,
            attribute=attribute,
            batch_size=2,
            update_interval=10,
        )
        assert set(np.where(visited)[0].tolist()) == {0, 1, 2}  # the 3 most influential

    def test_uncapped_attributes_every_feature(self):
        # multi-layer: greedy != global top-N when capped, but cap=all must attribute ALL.
        logit_edges = [(4, 0, 5.0), (4, 1, 1.0)]
        oracle = {0: [(0, 2, 10.0)], 1: [(1, 3, 10.0)], 2: [], 3: []}
        store = list(logit_edges)

        def get_edges():
            return _edge_arrays(store)

        def attribute(chunk):
            for f in chunk:
                store.extend(oracle[f])

        visited = _select_features_by_influence(
            5,
            np.array([0, 1, 2, 3]),
            np.array([4]),
            np.array([1.0]),
            target_cap=4,
            get_edges=get_edges,
            attribute=attribute,
            batch_size=2,
            update_interval=10,
        )
        assert set(np.where(visited)[0].tolist()) == {0, 1, 2, 3}  # all features attributed
