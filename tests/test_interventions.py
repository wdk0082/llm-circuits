"""Tests for feature-ablation logic (CPU, no model / GPU / network)."""

from __future__ import annotations

import torch

from llm_circuits.circuits.interventions import (
    AblationResult,
    FeatureAblation,
    ablation_logit_effect,
    ablation_prob_effect,
    ablations_to_dict,
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


class TestAblationsToDict:
    def test_grouping(self):
        abls = [
            FeatureAblation(3, 100),
            FeatureAblation(3, 7, position=2),
            FeatureAblation(5, 9),
        ]
        assert ablations_to_dict(abls) == {3: [(None, 100), (2, 7)], 5: [(None, 9)]}

    def test_empty(self):
        assert ablations_to_dict([]) == {}


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
