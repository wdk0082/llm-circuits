"""Unit tests for the verdict-carrying notebook helpers (DEVLOG_EXTRA §4.3).

The helpers live under ``notebooks/`` (task-specific, outside the package); this file
puts their pure logic — position parsing, the operand-grid classifier, the batched
direct-weight screen, the supernode %-readout — under pytest/CI. Everything here is
CPU-only and model-free (mock modules where weights are needed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import addition_helper as A
import multilingual_helper as M

# ---------------------------------------------------------------------------
# digit_token_positions (DEVLOG_EXTRA §1.1 regression)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Decode-only stub: id -> fixed string."""

    def __init__(self, toks: list[str]):
        self.toks = toks

    def decode(self, ids):
        return "".join(self.toks[i] for i in ids)


def _ids(n: int) -> torch.Tensor:
    return torch.arange(n).unsqueeze(0)


def test_digit_positions_teacher_forced_stops_at_equals():
    # <sink> calc :  ' ' 4  6  +  4  9  =  9   (teacher-forced ones prompt, 46+49)
    toks = ["<s>", "calc", ":", " ", "4", "6", "+", "4", "9", "=", "9"]
    pos = A.digit_token_positions(_FakeTokenizer(toks), _ids(len(toks)), 46, 49)
    assert pos["a_digits"] == [4, 5]
    assert pos["b_digits"] == [7, 8]  # pre-fix this was [8, 10]: forced answer digit kept
    assert pos["eq"] == [9]
    assert pos["plus"] == [6]


def test_digit_positions_plain_prompt():
    toks = ["<s>", "calc", ":", " ", "4", "6", "+", "4", "9", "="]
    pos = A.digit_token_positions(_FakeTokenizer(toks), _ids(len(toks)), 46, 49)
    assert pos["a_digits"] == [4, 5]
    assert pos["b_digits"] == [7, 8]


def test_digit_positions_multi_forced_digits():
    # 99+99=198 with two forced digits after '='
    toks = ["<s>", "calc", ":", " ", "9", "9", "+", "9", "9", "=", "1", "9"]
    pos = A.digit_token_positions(_FakeTokenizer(toks), _ids(len(toks)), 99, 99)
    assert pos["b_digits"] == [7, 8]


# ---------------------------------------------------------------------------
# periodicity_report band criterion (basis of the magnitude-verdict upgrade)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid_axes():
    a = np.repeat(np.arange(100)[:, None], 100, axis=1)
    b = np.repeat(np.arange(100)[None, :], 100, axis=0)
    return a, b, list(range(100)), list(range(100))


def test_band_a_with_weak_cross_arm(grid_axes):
    a, b, av, bv = grid_axes
    # max-composition (features saturate rather than add at the crossing — an additive
    # arm would create a hotspot that resets the 0.7 core): weak arm at 0.6 of the band
    # sits above the 0.5 activity threshold but below the 0.7 core -> band detected
    g = np.maximum(
        np.exp(-((a - 46.0) ** 2) / (2 * 4**2)), 0.6 * np.exp(-((b - 46.0) ** 2) / (2 * 4**2))
    )
    assert A.periodicity_report(g, av, bv)["label"] == "band-a(~46)"


def test_band_b_razor(grid_axes):
    a, b, av, bv = grid_axes
    g = np.maximum(
        np.exp(-((b - 49.0) ** 2) / (2 * 1.5**2)), 0.5 * np.exp(-((a - 49.0) ** 2) / (2 * 3**2))
    )
    assert A.periodicity_report(g, av, bv)["label"] == "band-b(~49)"


def test_mod10_stripe_not_stolen_by_band(grid_axes):
    a, _, av, bv = grid_axes
    assert A.periodicity_report((a % 10 == 6).astype(float), av, bv)["label"] == "mod10-a(r6)"


def test_equal_arm_cross_stays_out(grid_axes):
    a, b, av, bv = grid_axes
    g = np.exp(-((a - 46.0) ** 2) / (2 * 3**2)) + np.exp(-((b - 46.0) ** 2) / (2 * 3**2))
    assert not str(A.periodicity_report(g, av, bv)["label"]).startswith("band")


def test_uniform_and_diag(grid_axes):
    a, b, av, bv = grid_axes
    assert A.periodicity_report(np.ones((100, 100)), av, bv)["label"] == "mixed"
    diag = (np.abs(a + b - 95) <= 3).astype(float)
    assert A.periodicity_report(diag, av, bv)["label"] == "magnitude-diag"


# ---------------------------------------------------------------------------
# direct_token_weights == direct_logit_effect (batched vs reference)
# ---------------------------------------------------------------------------


class _FakeTC:
    def __init__(self, w):
        self.w = w

    def _get_decoder_vectors(self, idx):
        return self.w[idx]


def _mock_model_and_tc(d_model=16, vocab=50, d_t=32, n_layers=3, seed=0):
    torch.manual_seed(seed)
    norm = SimpleNamespace(weight=torch.randn(d_model))
    head = SimpleNamespace(weight=torch.randn(vocab, d_model))
    model = SimpleNamespace(
        get_submodule=lambda name: {"model.norm": norm, "lm_head": head}[name],
        parameters=lambda: iter([norm.weight]),
    )
    tc = SimpleNamespace(
        transcoders={L: _FakeTC(torch.randn(d_t, d_model)) for L in range(n_layers)}
    )
    return model, tc


def test_direct_token_weights_matches_reference():
    model, tc = _mock_model_and_tc()
    feats = [(0, 3), (1, 7), (2, 31), (0, 0)]
    out = A.direct_token_weights(model, tc, feats, token_id=9)
    for L, i in feats:
        ref = M.direct_logit_effect(model, tc, L, i, {"t": 9})["t"]
        assert out[(L, i)] == pytest.approx(ref, abs=1e-4)


# ---------------------------------------------------------------------------
# supernode_readout_pct (mean of individual ratios; ref modes; skip counting)
# ---------------------------------------------------------------------------


def _fake_result(seq=4, d_t=8, layers=(0, 1)):
    base = {L: torch.zeros(seq, d_t) for L in layers}
    abl = {L: torch.zeros(seq, d_t) for L in layers}
    return SimpleNamespace(baseline_features=base, ablated_features=abl)


def test_readout_mean_of_individual_ratios():
    res = _fake_result()
    fin = 3
    # feature (0,1): 10 -> 5 (50%); feature (1,2): 4 -> 8 (200%) -> mean 125, NOT
    # ratio-of-sums (13/14 = 92.9%)
    res.baseline_features[0][fin, 1] = 10.0
    res.ablated_features[0][fin, 1] = 5.0
    res.baseline_features[1][fin, 2] = 4.0
    res.ablated_features[1][fin, 2] = 8.0
    out = M.supernode_readout_pct(res, [(0, 1), (1, 2)], fin)
    assert out["mean_pct"] == pytest.approx(125.0)
    assert out["n_used"] == 2 and out["n_skipped"] == 0


def test_readout_zero_reference_skipped_and_stored_ref():
    res = _fake_result()
    fin = 3
    res.ablated_features[0][fin, 1] = 3.0  # baseline 0 -> skipped under ref="baseline"
    out = M.supernode_readout_pct(res, [(0, 1)], fin)
    assert out["mean_pct"] is None and out["n_skipped"] == 1
    # ref="stored" uses the node's carried act as the denominator
    out2 = M.supernode_readout_pct(res, [(0, 1, 6.0)], fin, ref="stored")
    assert out2["mean_pct"] == pytest.approx(50.0)


def test_readout_final_position_and_missing_layer():
    res = _fake_result()
    res.baseline_features[0][3, 1] = 2.0
    res.ablated_features[0][3, 1] = 1.0
    out = M.supernode_readout_pct(res, [(0, 1)], "final")
    assert out["mean_pct"] == pytest.approx(50.0)
    with pytest.raises(KeyError):
        M.supernode_readout_pct(res, [(5, 0)], "final")
