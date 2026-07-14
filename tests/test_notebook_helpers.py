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
    # <sink> calc :  ' ' 4  6  +  4  9  =  9   (teacher-forced ones prompt, 46+49):
    # the forced answer digit after '=' must NOT leak into b_digits
    toks = ["<s>", "calc", ":", " ", "4", "6", "+", "4", "9", "=", "9"]
    pos = A.digit_token_positions(_FakeTokenizer(toks), _ids(len(toks)), 46, 49)
    assert pos["a_digits"] == [4, 5]
    assert pos["b_digits"] == [7, 8]
    assert pos["eq"] == 9
    assert pos["plus"] == 6


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


def test_digit_positions_mismatch_raises():
    toks = ["<s>", "calc", ":", " ", "4", "7", "+", "4", "9", "="]
    with pytest.raises(ValueError, match="operand tokens mismatch"):
        A.digit_token_positions(_FakeTokenizer(toks), _ids(len(toks)), 46, 49)


# ---------------------------------------------------------------------------
# classify_grid: operand-plot receptive-field families (v3 clean-room classifier)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid_axes():
    a = np.repeat(np.arange(100)[:, None], 100, axis=1)
    b = np.repeat(np.arange(100)[None, :], 100, axis=0)
    return a, b


def test_lookup_lattice_and_mod10_families(grid_axes):
    a, b = grid_axes
    lat = ((a % 10 == 6) & (b % 10 == 9)).astype(float) * 5
    assert A.classify_grid(lat)[0] == "lookup(a%10=6,b%10=9)"
    assert A.classify_grid((a % 10 == 6).astype(float))[0] == "mod10-a(r6)"
    assert A.classify_grid(((a + b) % 10 == 5).astype(float))[0] == "mod10-sum(r5)"


def test_band_a_with_weak_cross_arm(grid_axes):
    a, b = grid_axes
    # max-composition (features saturate rather than add at the crossing): the weak
    # perpendicular arm must not break the magnitude-band call
    g = np.maximum(
        np.exp(-((a - 46.0) ** 2) / (2 * 4**2)), 0.6 * np.exp(-((b - 46.0) ** 2) / (2 * 4**2))
    )
    assert A.classify_grid(g)[0].startswith("band-a(~4")


def test_band_b_razor(grid_axes):
    a, b = grid_axes
    g = np.maximum(
        np.exp(-((b - 49.0) ** 2) / (2 * 1.5**2)), 0.5 * np.exp(-((a - 49.0) ** 2) / (2 * 3**2))
    )
    assert A.classify_grid(g)[0].startswith("band-b(~4")


def test_equal_arm_cross_not_a_band(grid_axes):
    a, b = grid_axes
    g = np.exp(-((a - 46.0) ** 2) / (2 * 3**2)) + np.exp(-((b - 46.0) ** 2) / (2 * 3**2))
    assert not A.classify_grid(g)[0].startswith("band")


def test_uniform_and_sum_band(grid_axes):
    a, b = grid_axes
    assert A.classify_grid(np.ones((100, 100)))[0] == "mixed"
    diag = (np.abs(a + b - 95) <= 3).astype(float)
    assert A.classify_grid(diag)[0].startswith("sum-band(~9")
    assert A.classify_grid(np.zeros((100, 100)))[0] == "dead"


def test_cross_region_and_lookup_not_stolen(grid_axes):
    a, b = grid_axes
    # exact-value cross: fires when EITHER operand is 46 (the paper's "36"/"59"
    # input-identity features) — must beat the mod10-a residue class
    g = ((a == 46) | (b == 46)).astype(float)
    assert A.classify_grid(g)[0] == "cross(46)"
    # 2-D localized blob without repetition: the low-precision magnitude-lookup class
    g2 = np.exp(-((a - 46.0) ** 2) / (2 * 4**2) - ((b - 49.0) ** 2) / (2 * 4**2))
    assert A.classify_grid(g2)[0] == "region(~46,~49)"
    # a smeared modular lattice (two adjacent b residues) stays in the lookup family
    g3 = (((a % 10) == 6) & (((b % 10) == 9) | ((b % 10) == 8))).astype(float)
    assert A.classify_grid(g3)[0].startswith("lookup(a%10=6")
    # the studied pair's cell as a fraction of max
    assert A.on_pair_fraction(g3, (46, 49)) == pytest.approx(1.0)
    assert A.on_pair_fraction(np.zeros((100, 100)), (46, 49)) == 0.0


# ---------------------------------------------------------------------------
# digit_direct_weights == direct_logit_effect (batched digits vs reference)
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


class _DigitTokenizer(_FakeTokenizer):
    """Callable stub: text "3" -> input_ids [3] (digit tokens are ids 0..9)."""

    def __call__(self, text, **_kw):
        return SimpleNamespace(input_ids=[self.toks.index(text)])


def test_digit_direct_weights_matches_reference():
    model, tc = _mock_model_and_tc()
    tok = _DigitTokenizer([str(d) for d in range(10)])
    feats = [(0, 3), (1, 7), (2, 31), (0, 0)]
    out = A.digit_direct_weights(model, tc, feats, tok)
    for L, i in feats:
        assert len(out[(L, i)]) == 10
        for d in (0, 5, 9):
            ref = M.direct_logit_effect(model, tc, L, i, {"t": d})["t"]
            assert out[(L, i)][d] == pytest.approx(ref, abs=1e-4)


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


# ---------------------------------------------------------------------------
# Constrained-patching plumbing (DEVLOG_EXTRA §3.1)
# ---------------------------------------------------------------------------


def _sn(name, members, role="source"):
    return {
        "name": name,
        "role": role,
        "graph": "g",
        "position": "member-positions",
        "members": members,
    }


def test_supernode_suppress_ivs_member_positions_and_fallback():
    sn = _sn(
        "antonym (multilingual)",
        [
            {
                "layer": 7,
                "feature": 10,
                "positions": {"antonym_en": 6},
                "acts": {"antonym_en": 3.0},
            },
            {"layer": 34, "feature": 20, "positions": {"raw_antonym_en": 1}, "acts": {}},
        ],
    )
    ivs = M.supernode_suppress_ivs(sn, "antonym_en", mult=-5.0, fallback_pos=28)
    # member with a recorded position steers there; the raw-only member falls back to final
    assert [(iv.layer, iv.feature_idx, iv.position) for iv in ivs] == [(7, 10, 6), (34, 20, 28)]
    assert all(iv.m == pytest.approx(-6.0) and iv.value is None for iv in ivs)  # M=-5 -> m=-6


def test_supernode_inject_ivs_span_values_and_skip():
    sn = _sn(
        "synonym (multilingual)",
        [
            {"layer": 19, "feature": 3, "acts": {"synonym_en": 5.0, "raw_synonym_en": 10.0}},
            {"layer": 34, "feature": 4, "acts": {"raw_synonym_en": 32.0}},  # no chat act -> skipped
        ],
        role="donor",
    )
    ivs, used = M.supernode_inject_ivs(sn, "synonym_en", [6, 7], mult=6.0)
    # one iv per span position, absolute value = 6x the DONOR-graph act; raw-only member skipped
    assert [(iv.layer, iv.feature_idx, iv.position, iv.value) for iv in ivs] == [
        (19, 3, 6, pytest.approx(30.0)),
        (19, 3, 7, pytest.approx(30.0)),
    ]
    assert used == [(19, 3, 5.0)]


def test_swap_ivs_fn_ramp_hits_endpoint_pair():
    src = _sn("s", [{"layer": 2, "feature": 1, "positions": {"g": 3}, "acts": {"g": 2.0}}])
    don = _sn("d", [{"layer": 6, "feature": 9, "acts": {"dg": 5.0}}], role="donor")
    fn = M.swap_ivs_fn(src, "g", 4, don, "dg", [4], kind="operation")
    assert fn(0.0) == []
    ivs = fn(6.0)  # s = don_max -> the paper endpoint pair (-5x source, +6x donor)
    assert ivs[0].m == pytest.approx(-6.0) and ivs[0].position == 3
    assert ivs[1].value == pytest.approx(30.0) and ivs[1].position == 4
    half = fn(3.0)  # proportional ramp: source_mult = 1 + (-6)(3/6) = -2 -> m = -3
    assert half[0].m == pytest.approx(-3.0) and half[1].value == pytest.approx(15.0)


def test_swap_ivs_fn_strengths_override_extends_same_ramp():
    src = _sn("s", [{"layer": 2, "feature": 1, "positions": {"g": 3}, "acts": {"g": 2.0}}])
    don = _sn("d", [{"layer": 6, "feature": 9, "acts": {"dg": 5.0}}], role="donor")
    # (-14, 15) preserves the operand coupling (src_max-1)/don_max = -1, so the ladder
    # passes through the paper endpoint (-0.5x, +1.5x) at s = 1.5 on its way to 15
    fn = M.swap_ivs_fn(src, "g", 4, don, "dg", [4], kind="operand", strengths=(-14.0, 15.0))
    at_paper = fn(1.5)
    assert at_paper[0].m == pytest.approx(-1.5)  # M_paper = -0.5 -> m = -1.5
    assert at_paper[1].value == pytest.approx(7.5)  # +1.5x the 5.0 stored act
    at_end = fn(15.0)
    assert at_end[0].m == pytest.approx(-15.0) and at_end[1].value == pytest.approx(75.0)


def _v3_sn(name, members, *, graph="ones", position="final", role="source"):
    return {"name": name, "role": role, "graph": graph, "position": position, "members": members}


def test_addition_suppress_ivs_positions_and_m():
    sn = _v3_sn(
        "input _6",
        [
            {"layer": 7, "feature": 10, "position": 5, "act": 3.0},
            {"layer": 34, "feature": 20, "position": "final", "act": 1.0},
            {"layer": 12, "feature": 30, "position": None, "act": 2.0},
        ],
    )
    ivs = A.suppress_ivs(sn, mult=-1.0, final_pos=9)
    # recorded position kept; "final"/None resolve to final_pos; M=-1 -> m=-2
    assert [(iv.layer, iv.feature_idx, iv.position) for iv in ivs] == [
        (7, 10, 5),
        (34, 20, 9),
        (12, 30, 9),
    ]
    assert all(iv.m == pytest.approx(-2.0) and iv.value is None for iv in ivs)


def test_addition_inject_ivs_values_and_skip():
    sn = _v3_sn(
        "lookup (9,9) donors",
        [
            {"layer": 19, "feature": 3, "position": "final", "act": 5.0},
            {"layer": 22, "feature": 4, "position": "final", "act": 0.0},  # no act -> skipped
        ],
        graph="donor",
        role="donor",
    )
    ivs, used = A.inject_ivs(sn, mult=2.0, positions=[9, 10])
    assert [(iv.layer, iv.feature_idx, iv.position, iv.value) for iv in ivs] == [
        (19, 3, 9, pytest.approx(10.0)),
        (19, 3, 10, pytest.approx(10.0)),
    ]
    assert used == [(19, 3, 5.0)]


def test_layer_sweep_result_pick_properties():
    from llm_circuits.circuits.interventions import LayerSweepResult

    sweep = LayerSweepResult(
        end_layers=[10, 11, 12, 13],
        logits=[5.0, 2.0, 4.0, 2.0],  # min at ell=11 (first of the tie 11/13)
        probs=[0.1, 0.3, 0.9, 0.9],  # max at ell=12 (first of the tie 12/13)
        baseline_logit=6.0,
        baseline_prob=0.5,
        token_id=42,
        position=-1,
    )
    assert sweep.best_end_layer == 11
    assert sweep.most_promoting_end_layer == 12


def test_readout_pinned_rows_dropped_under_constrained_patching():
    res = _fake_result(layers=(0, 1))
    fin = 3
    res.baseline_features[0][fin, 1] = 10.0
    res.ablated_features[0][fin, 1] = 5.0  # L0 <= ell -> pinned, excluded
    res.baseline_features[1][fin, 2] = 4.0
    res.ablated_features[1][fin, 2] = 8.0  # L1 > ell -> the only used row
    out = M.supernode_readout_pct(res, [(0, 1), (1, 2)], fin, patch_end_layer=0)
    assert out["mean_pct"] == pytest.approx(200.0)
    assert out["n_used"] == 1 and out["n_pinned"] == 1
    assert out["per_feature"]["L0f1"] == "pinned"
    # pinned check runs before the layer-coverage check (no KeyError for pinned layers
    # missing from readout_layers)
    out2 = M.supernode_readout_pct(res, [(9, 0)], fin, patch_end_layer=9)
    assert out2["mean_pct"] is None and out2["n_pinned"] == 1


def test_steer_report_patch_end_layer_passthrough_and_pinned_readout(monkeypatch):
    captured = {}

    def fake_run(model, tc, input_ids, ivs, **kw):
        captured["patch_end_layer"] = kw.get("patch_end_layer")
        captured["readout_layers"] = kw.get("readout_layers")
        base = {L: torch.zeros(4, 8) for L in (2, 5)}
        abl = {L: torch.zeros(4, 8) for L in (2, 5)}
        base[5][3, 1] = 2.0
        abl[5][3, 1] = 1.0
        logits = torch.zeros(4, 16)
        return SimpleNamespace(
            baseline_logits=logits,
            ablated_logits=logits,
            baseline_features=base,
            ablated_features=abl,
        )

    monkeypatch.setattr(A, "run_feature_intervention", fake_run)
    tok = _DigitTokenizer([str(d) for d in range(10)] + [""] * 6)
    sn = _v3_sn(
        "down",
        [
            {"layer": 2, "feature": 1, "position": 3, "act": 1.0},  # L2 <= ell -> pinned
            {"layer": 5, "feature": 1, "position": 3, "act": 1.0},
        ],
        role="readout",
    )
    rep = A.steer_report(
        None,
        None,
        tok,
        torch.zeros(1, 4, dtype=torch.long),
        [],
        patch_end_layer=3,
        readout_sns={"down": (sn, "baseline")},
    )
    assert captured["patch_end_layer"] == 3
    assert captured["readout_layers"] == [5]  # pinned layers not requested
    assert rep["patch_end_layer"] == 3
    ro = rep["readouts"]["down"]
    assert ro["per_feature"]["L2f1"] == "pinned" and ro["n_pinned"] == 1
    assert ro["per_feature"]["L5f1"] == pytest.approx(50.0)
    assert ro["mean_pct"] == pytest.approx(50.0)
    # uniform logits -> renormalized digit distribution is uniform -> smear width 10
    assert rep["width_before"] == pytest.approx(10.0)
    assert len(rep["digits_after"]) == 10


def test_supernode_swap_sweep_constrained_at_fixed_ell(monkeypatch):
    calls = []

    def fake_run(model, tc, input_ids, ivs, **kw):
        calls.append((len(ivs), kw.get("patch_end_layer")))
        k = float(len(calls))  # step counter: baseline falls, expected rises with strength
        logits = torch.zeros(4, 16)
        logits[-1, 1] = 3.0 - k
        logits[-1, 2] = k
        return SimpleNamespace(ablated_logits=logits, baseline_logits=logits)

    monkeypatch.setattr(M, "run_feature_intervention", fake_run)
    src = _sn("s", [{"layer": 2, "feature": 1, "positions": {"g": 3}, "acts": {"g": 2.0}}])
    don = _sn("d", [{"layer": 6, "feature": 9, "acts": {"dg": 5.0}}], role="donor")
    fn = M.swap_ivs_fn(src, "g", 4, don, "dg", [4], kind="operation")
    tok = _FakeTokenizer([str(d) for d in range(16)])
    r = M.supernode_swap_sweep(
        None,
        None,
        torch.zeros(1, 4, dtype=torch.long),
        fn,
        tok,
        kind="operation",
        baseline_token=1,
        expected_token=2,
        patch_end_layer=7,
        n_steps=4,
    )
    # s=0 runs the clean forward (no patching); every other step is constrained at ell=7
    assert calls[0] == (0, None) and all(c == (2, 7) for c in calls[1:])
    assert r["patch_end_layer"] == 7 and r["strengths"][-1] == pytest.approx(6.0)
    assert r["crossover"] is not None


def test_supernode_readout_member_positions_pinned_and_stored_ref():
    res = _fake_result(layers=(0, 1))
    # member A reads at ITS OWN node position (2), member B at the final fallback (3)
    res.baseline_features[1][2, 1] = 10.0
    res.ablated_features[1][2, 1] = 5.0
    res.baseline_features[1][3, 2] = 4.0
    res.ablated_features[1][3, 2] = 8.0
    sn = _sn(
        "r",
        [
            {"layer": 1, "feature": 1, "positions": {"g": 2}, "acts": {"g": 10.0}},
            {"layer": 1, "feature": 2, "positions": {}, "acts": {"other": 16.0}},
            {"layer": 0, "feature": 5, "positions": {"g": 3}, "acts": {"g": 1.0}},
        ],
        role="readout",
    )
    out = M.supernode_readout(res, sn, "g", final_pos=3, patch_end_layer=0)
    # L0 member pinned; L1 members: 5/10=50% @p2 and 8/4=200% @p3 -> mean 125%
    assert out["mean_pct"] == pytest.approx(125.0)
    assert out["n_used"] == 2 and out["n_pinned"] == 1
    assert out["per_feature"]["L0f5@p3"] == "pinned"
    # ref_graph: stored act there, falling back to the member's max stored act
    out2 = M.supernode_readout(res, sn, "g", final_pos=3, ref_graph="g", patch_end_layer=0)
    assert out2["per_feature"]["L1f1@p2"] == pytest.approx(50.0)  # 5 / stored 10
    assert out2["per_feature"]["L1f2@p3"] == pytest.approx(50.0)  # 8 / fallback max 16


def test_load_supernodes_multilingual_global_disjointness(tmp_path):
    import json

    doc = {
        "approved": True,
        "supernodes": [
            _sn("a", [{"layer": 1, "feature": 2, "acts": {}, "positions": {}}]),
            _sn("b", [{"layer": 1, "feature": 2, "acts": {}, "positions": {}}]),
        ],
    }
    p = tmp_path / "ml.json"
    p.write_text(json.dumps(doc))
    # same (layer, feature) in two supernodes — even with different graph/position keys —
    # violates the multilingual file invariant (disjoint FEATURE sets)
    with pytest.raises(ValueError, match="disjoint feature sets"):
        M.load_supernodes(p)
    doc["supernodes"][1]["members"][0]["feature"] = 3
    p.write_text(json.dumps(doc))
    out = M.load_supernodes(p)
    assert set(out) == {"a", "b"}
    doc["approved"] = False
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="approved=false"):
        M.load_supernodes(p)


def test_cjk_font_registered_for_zh_panels():
    # multilingual_helper's import-time font block must yield a usable CJK family
    # (system font or the mplfonts-bundled Noto Sans CJK SC) so ZH sweep panels don't
    # render tofu. mplfonts is in the notebook dependency group, present under CI's
    # --all-groups install.
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fams = plt.rcParams["font.family"]
    assert any("CJK" in f for f in fams), f"no CJK family in font.family: {fams}"
    resolved = font_manager.findfont(font_manager.FontProperties(family="Noto Sans CJK SC"))
    assert "NotoSansCJK" in resolved.replace(" ", "")


# ---------------------------------------------------------------------------
# Addition supernode loader (v3: selection lives in the reviewed artifact)
# ---------------------------------------------------------------------------


def _sn_doc(**over):
    doc = {
        "approved": True,
        "supernodes": [
            {
                "name": "a",
                "graph": "ones",
                "position": "final",
                "members": [
                    {"layer": 1, "feature": 2, "position": 5, "act": 3.0, "review": "approved"},
                ],
            }
        ],
    }
    doc.update(over)
    return doc


def test_load_supernodes_happy_path_and_position_resolution(tmp_path):
    import json

    p = tmp_path / "sn.json"
    p.write_text(json.dumps(_sn_doc()))
    out = A.load_supernodes(p)
    m = out["a"]["members"][0]
    assert (m["layer"], m["feature"], m["act"]) == (1, 2, 3.0)
    # member position wins; "final"/None fall back through the supernode's
    assert A.resolve_position(out["a"], m, final_pos=9) == 5
    assert A.resolve_position(out["a"], {"layer": 1, "feature": 3}, final_pos=9) == 9


def test_load_supernodes_refuses_unapproved_rejected_and_overlap(tmp_path):
    import json

    p = tmp_path / "sn.json"
    p.write_text(json.dumps(_sn_doc(approved=False)))
    with pytest.raises(ValueError, match="approved=false"):
        A.load_supernodes(p)

    doc = _sn_doc()
    doc["supernodes"][0]["members"][0]["review"] = "rejected"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="rejected member"):
        A.load_supernodes(p)

    # same (layer, feature) twice on the SAME graph -> refused
    doc = _sn_doc()
    doc["supernodes"].append(dict(doc["supernodes"][0], name="b"))
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="in both"):
        A.load_supernodes(p)

    # ... but the same feature on a DIFFERENT graph is legitimate (reuse-context groups)
    doc = _sn_doc()
    doc["supernodes"].append(dict(doc["supernodes"][0], name="b", graph="reuse"))
    p.write_text(json.dumps(doc))
    assert set(A.load_supernodes(p)) == {"a", "b"}
