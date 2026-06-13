"""Tests for the batched edge-weight math in attribution_graph (CPU, no model)."""

from __future__ import annotations

import inspect

import torch

from llm_circuits.circuits.attribution_graph import (
    _batched_edge_weights,
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


def test_build_attribution_graph_exposes_min_edge_weight():
    params = inspect.signature(build_attribution_graph).parameters
    assert "min_edge_weight" in params
    assert params["min_edge_weight"].default == 0.0
