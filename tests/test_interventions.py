"""Tests for feature-ablation logic (CPU, no model / GPU / network)."""

from __future__ import annotations

import torch

from llm_circuits.circuits.interventions import (
    AblationResult,
    FeatureIntervention,
    LayerSweepResult,
    ablation_logit_effect,
    ablation_prob_effect,
    steer,
)
from llm_circuits.circuits.local_replacement_model import _apply_ablations


class TestApplyAblations:
    def test_none_is_identity(self):
        feats = torch.randn(4, 6)
        assert _apply_ablations(feats, None) is feats
        assert _apply_ablations(feats, []) is feats

    def test_zeroes_all_positions_when_position_none(self):
        feats = torch.ones(4, 6)
        out = _apply_ablations(feats, [(None, 2)])
        assert torch.all(out[:, 2] == 0.0)
        # other features untouched
        assert torch.all(out[:, [0, 1, 3, 4, 5]] == 1.0)

    def test_zeroes_single_cell(self):
        feats = torch.ones(4, 6)
        out = _apply_ablations(feats, [(1, 3)])
        assert out[1, 3] == 0.0
        # only that one cell changed
        assert out.sum() == feats.sum() - 1.0

    def test_not_in_place(self):
        feats = torch.ones(4, 6)
        _apply_ablations(feats, [(None, 0)])
        assert torch.all(feats == 1.0)  # original untouched

    def test_multiple_ablations(self):
        feats = torch.ones(3, 5)
        out = _apply_ablations(feats, [(0, 0), (None, 4)])
        assert out[0, 0] == 0.0
        assert torch.all(out[:, 4] == 0.0)

    def test_batched_leading_dim(self):
        feats = torch.ones(1, 4, 6)  # (batch, seq, d_transcoder)
        out = _apply_ablations(feats, [(2, 5)])
        assert out[0, 2, 5] == 0.0
        assert out.sum() == feats.sum() - 1.0


class TestAblationLogitEffect:
    def _result(self) -> AblationResult:
        base = torch.zeros(3, 10)
        abl = torch.zeros(3, 10)
        # at last position, token 5 drops by 2.0, token 1 rises by 0.5
        abl[-1, 5] = -2.0
        abl[-1, 1] = 0.5
        return AblationResult(baseline_logits=base, ablated_logits=abl)

    def test_default_last_position(self):
        eff = ablation_logit_effect(self._result(), [5, 1, 2])
        assert eff[5] == -2.0
        assert eff[1] == 0.5
        assert eff[2] == 0.0

    def test_logit_delta_property(self):
        r = self._result()
        assert torch.allclose(r.logit_delta, r.ablated_logits - r.baseline_logits)

    def test_explicit_position(self):
        r = self._result()
        # position 0 has no changes
        assert ablation_logit_effect(r, [5, 1], position=0) == {5: 0.0, 1: 0.0}


class TestAblationProbEffect:
    def test_baseline_and_ablated_probs(self):
        # Baseline: token 0 dominates (logit 10 vs 0). Ablated: uniform over 4.
        base = torch.zeros(2, 4)
        base[-1] = torch.tensor([10.0, 0.0, 0.0, 0.0])
        abl = torch.zeros(2, 4)  # all zeros -> uniform softmax
        r = AblationResult(baseline_logits=base, ablated_logits=abl)
        bp, ap = ablation_prob_effect(r, [0])[0]
        assert bp > 0.99  # token 0 ~certain at baseline
        assert abs(ap - 0.25) < 1e-5  # uniform after ablation

    def test_probs_are_valid_distribution(self):
        base = torch.randn(3, 5)
        abl = torch.randn(3, 5)
        r = AblationResult(baseline_logits=base, ablated_logits=abl)
        effects = ablation_prob_effect(r, [0, 1, 2, 3, 4])
        for bp, ap in effects.values():
            assert 0.0 <= bp <= 1.0
            assert 0.0 <= ap <= 1.0


class TestFeatureIntervention:
    # M convention: new activation = (1 + m) * clean; m=0 no change, m=-1 ablate, m=-2 flip.
    def test_default_is_ablation(self):
        # Default m=-1 -> target 0 (ablation) regardless of clean activation.
        assert FeatureIntervention(3, 100).target(5.0) == 0.0

    def test_explicit_value(self):
        assert FeatureIntervention(3, 100, value=2.5).target(5.0) == 2.5

    def test_m_convention(self):
        assert FeatureIntervention(3, 100, m=0.0).target(5.0) == 5.0  # no change
        assert FeatureIntervention(3, 100, m=-1.0).target(5.0) == 0.0  # ablate
        assert FeatureIntervention(3, 100, m=1.0).target(4.0) == 8.0  # double
        assert FeatureIntervention(3, 100, m=-2.0).target(3.0) == -3.0  # flip

    def test_value_takes_precedence_over_m(self):
        assert FeatureIntervention(3, 100, value=1.0, m=-2.0).target(5.0) == 1.0

    def test_helpers(self):
        assert steer(7, 9, m=0.5, position=2) == FeatureIntervention(7, 9, position=2, m=0.5)
        assert steer(7, 9, m=-1.0).target(3.0) == 0.0  # m=-1 -> ablate
        ns = steer(7, 9, m=-2.0, position=2)
        assert ns == FeatureIntervention(7, 9, position=2, m=-2.0)
        assert ns.target(3.0) == -3.0  # m=-2 -> flip sign


class TestLayerSweepResult:
    def _res(self):
        # baseline logit 5.0; end layers 20..23 with decreasing logits then a rebound
        return LayerSweepResult(
            end_layers=[20, 21, 22, 23],
            logits=[2.0, -1.0, -3.0, 0.5],
            probs=[0.40, 0.20, 0.05, 0.30],
            baseline_logit=5.0,
            baseline_prob=0.80,
            token_id=42,
            position=-1,
        )

    def test_delta_logits_and_probs(self):
        import pytest

        r = self._res()
        assert r.delta_logits == pytest.approx([-3.0, -6.0, -8.0, -4.5])
        assert r.delta_probs == pytest.approx([-0.40, -0.60, -0.75, -0.50])

    def test_best_end_layer_is_max_suppression(self):
        # layer 22 has the lowest logit (-3.0) -> largest suppression
        assert self._res().best_end_layer == 22
