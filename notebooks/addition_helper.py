"""Addition-circuit helpers for reproducing the paper's Addition dive on Qwen3-4b (v3).

https://transformer-circuits.pub/2025/attribution-graphs/biology.html (Addition section)

The paper studies ``calc: 36+59=`` -> ``95`` on Claude 3.5 Haiku and finds a circuit of
parallel pathways meeting at the answer: **input-digit** features (``_6``, ``_9``: the
operands' ones digits), **low-precision magnitude** features (~36, ~59, and "sum near
92"), a **lookup-table** feature (``_6+_9``) carrying the high-precision fact "ones
digits 6 and 9 sum to ...5", and **sum** features (``sum = _5``, ``sum ~95``) that write
the answer. Feature identities are established with **operand plots**: a feature's
activation over all 10,000 prompts ``calc: a+b=`` (a, b in 0..99), read at the ``=``
token — diagonal stripes = sum features, rows/columns = operand features, isolated
lattice points = lookup tables, mod-10 lattices = modular structure, smeared versions =
low precision. Interventions: suppressing supernodes weakens their downstream partners
and moves the answer; substituting ``_6+_9`` with ``_9+_9`` retargets the ones digit
5 -> 8; the same lookup features fire (and act) in non-arithmetic contexts (journal
volume/year citations, running totals); asked "Briefly, how did you get that?" the model
describes the carry algorithm it does not use.

**v3 is a clean-room rewrite** (2026-07-14): the paper is the spec and
``multilingual_helper.py`` the repo template; no code is taken from the v2 addition
helper. Selection lives in the reviewed ``supernodes/addition_<size>.json``
(explorer-export ingest with operand-grid evidence — see ``build_supernodes.py``);
:func:`load_supernodes` refuses anything unapproved. Established environment facts carry
over as constraints, not code:

* Qwen3 tokenizes numbers **digit by digit**, so "95" is emitted as "9" then "5" and the
  paper's single answer circuit splits into a **first-digit** graph (``...=`` -> "9") and
  a teacher-forced **ones-digit** graph (``...=9`` -> "5"). Operand plots probe both
  moments (:func:`operand_grids`).
* Qwen3-4b does not answer every pair correctly (the paper's 36+59 included) —
  :func:`pick_studied_pair` selects the nearest correct pair of the same ones-digit
  class so the ``_6+_9`` story maps over.
* Steering strengths: the paper's multiplicative ``M`` is our additive ``m = M - 1``
  (``FeatureIntervention``: new activation = ``(1 + m) * clean``).

Heavy functions run the model — execute the notebook on a GPU node via
``hpc/run_addition_notebook.sbatch``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from llm_circuits.circuits.attribution_graph import build_attribution_graph
from llm_circuits.circuits.graph_explorer import render_graph_explorer_html
from llm_circuits.circuits.graph_pruning import graph_to_dict, prune_graph
from llm_circuits.circuits.interventions import (
    FeatureIntervention,
    run_feature_intervention,
    sweep_patch_end_layer,
)
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.transcoders.feature_labels import (
    load_feature_examples,
    load_feature_labels,
)
from llm_circuits.transcoders.registry import get_spec

N_BOS = 1  # raw prompts get one prepended sink token (Qwen3 has no BOS)

CALC_TEMPLATE = "calc: {a}+{b}="
PAPER_PAIR = (36, 59)  # the paper's studied problem; v3 falls back to a correct same-class pair

# The paper's academic-citation reuse context: a journal founded in a year ending like b,
# volume number like a, publication year = founded + volume — the same ones-digit fact
# the calc lookup feature carries, in prose. reuse_prompt(46, 49) ends "... appeared in 19"
# and the model should continue "9", "5" (1949 + 46 = 1995).
REUSE_TEMPLATE = (
    "The journal Annals of Applied Analysis was founded in {century}{b}. "
    "Counting one volume per year, volume {a} appeared in {century}"
)


def addition_prompt(a: int, b: int, suffix: str = "") -> str:
    """``calc: {a}+{b}=`` plus any teacher-forced answer digits in *suffix*."""
    return CALC_TEMPLATE.format(a=a, b=b) + suffix


def reuse_prompt(a: int, b: int, *, century: int = 19) -> str:
    """The citation-style reuse prompt for the pair (a, b); needs ``a + b < 100``."""
    if a + b >= 100:
        raise ValueError("reuse_prompt needs a+b < 100 so the century digits stay fixed")
    return REUSE_TEMPLATE.format(a=a, b=b, century=century)


# ---------------------------------------------------------------------------
# Tokenization (digit-by-digit; positions by construction)
# ---------------------------------------------------------------------------


def tokenize_raw(tokenizer, text: str, device) -> torch.Tensor:
    """Raw-completion tokenisation with a prepended special token (the attention sink).

    Same convention as ``multilingual_helper.tokenize_raw``: transcoders can't
    reconstruct position 0, so raw prompts get a special token prepended (eos/pad —
    Qwen3 has no BOS) and every downstream consumer keeps ``n_bos_tokens=1``.
    """
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    special = tokenizer.bos_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id
    ids = torch.cat([torch.tensor([[special]], dtype=ids.dtype), ids], dim=1)
    return ids.to(device)


def tokenize_calc(tokenizer, a: int, b: int, device, *, suffix: str = "") -> torch.Tensor:
    """Tokenize ``calc: a+b=<suffix>`` (sink token prepended) with a structure check."""
    ids = tokenize_raw(tokenizer, addition_prompt(a, b, suffix), device)
    toks = [tokenizer.decode([int(t)]) for t in ids[0]]
    digits = [t for t in toks if t.strip().isdigit()]
    if any(len(t.strip()) != 1 for t in digits):
        raise ValueError(f"expected digit-by-digit tokenization, got {toks!r}")
    return ids


def digit_token_positions(tokenizer, input_ids: torch.Tensor, a: int, b: int) -> dict:
    """Token positions of the operand digits and the ``=`` in a calc prompt.

    Returns ``{"a_digits": [...], "b_digits": [...], "plus": int, "eq": int}``. Works by
    locating the ``+`` and ``=`` tokens and counting the digit tokens between them; the
    digits are asserted against ``str(a)`` / ``str(b)`` so a surprising tokenization
    fails loudly rather than mis-anchoring interventions.
    """
    toks = [tokenizer.decode([int(t)]) for t in input_ids[0]]
    plus = next(i for i, t in enumerate(toks) if t.strip() == "+")
    eq = next(i for i, t in enumerate(toks) if t.strip() == "=")
    a_digits = list(range(plus - len(str(a)), plus))
    b_digits = list(range(plus + 1, eq))
    got_a = "".join(toks[i].strip() for i in a_digits)
    got_b = "".join(toks[i].strip() for i in b_digits)
    if got_a != str(a) or got_b != str(b):
        raise ValueError(f"operand tokens mismatch: {got_a!r}/{got_b!r} vs {a}/{b} in {toks!r}")
    return {"a_digits": a_digits, "b_digits": b_digits, "plus": plus, "eq": eq}


def _digit_token_ids(tokenizer) -> list[int]:
    """Token id of each single digit "0".."9" (asserted single-token)."""
    out = []
    for d in "0123456789":
        enc = tokenizer(d, add_special_tokens=False).input_ids
        if len(enc) != 1:
            raise ValueError(f"digit {d!r} is not a single token: {enc}")
        out.append(enc[0])
    return out


# ---------------------------------------------------------------------------
# §0 Behavior: accuracy over the full operand grid + studied-pair selection
# ---------------------------------------------------------------------------


@torch.no_grad()
def accuracy_grid(
    model, tokenizer, a_vals=None, b_vals=None, *, batch_size: int = 256, max_digits: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy-decode ``calc: a+b=`` for every pair; returns ``(correct, predicted)``.

    ``correct`` is a bool grid, ``predicted`` an int grid (-1 where the continuation
    is not a number). Batched with same-length grouping (operand digit counts vary).
    """
    a_vals = list(range(100)) if a_vals is None else list(a_vals)
    b_vals = list(range(100)) if b_vals is None else list(b_vals)
    device = next(model.parameters()).device
    digit_ids = torch.tensor(_digit_token_ids(tokenizer), device=device)

    prompts: dict[tuple[int, int], torch.Tensor] = {
        (a, b): tokenize_calc(tokenizer, a, b, device)[0] for a in a_vals for b in b_vals
    }
    by_len: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair, ids in prompts.items():
        by_len[ids.shape[0]].append(pair)

    correct = np.zeros((len(a_vals), len(b_vals)), dtype=bool)
    predicted = np.full((len(a_vals), len(b_vals)), -1, dtype=np.int64)
    ai = {a: i for i, a in enumerate(a_vals)}
    bi = {b: j for j, b in enumerate(b_vals)}

    for _n, pairs in sorted(by_len.items()):
        for k in range(0, len(pairs), batch_size):
            chunk = pairs[k : k + batch_size]
            ids = torch.stack([prompts[p] for p in chunk])
            digits: list[list[int]] = [[] for _ in chunk]
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            for _step in range(max_digits):
                logits = model(ids, attention_mask=torch.ones_like(ids)).logits[:, -1]
                nxt = logits.argmax(-1)
                is_digit = torch.isin(nxt, digit_ids)
                for r in range(len(chunk)):
                    if not bool(done[r]) and bool(is_digit[r]):
                        digits[r].append(int((nxt[r] == digit_ids).nonzero()))
                done |= ~is_digit
                if bool(done.all()):
                    break
                ids = torch.cat([ids, nxt[:, None]], dim=1)
            for r, (a, b) in enumerate(chunk):
                if digits[r]:
                    val = int("".join(str(d) for d in digits[r]))
                    predicted[ai[a], bi[b]] = val
                    correct[ai[a], bi[b]] = val == a + b
    return correct, predicted


def pick_pair_by_class(
    correct: np.ndarray, *, ones: tuple[int, int], near: tuple[int, int]
) -> tuple[int, int] | None:
    """Nearest correctly-answered two-digit pair with the given ones digits.

    Distance is L1 to *near*; ties break toward smaller (a, b). Assumes a 100x100 grid
    over 0..99. Returns None when no two-digit pair of that class is correct.
    """
    best, best_d = None, None
    for a in range(10 + ones[0], 100, 10):
        for b in range(10 + ones[1], 100, 10):
            if not correct[a, b]:
                continue
            d = abs(a - near[0]) + abs(b - near[1])
            if best_d is None or d < best_d or (d == best_d and (a, b) < best):
                best, best_d = (a, b), d
    return best


def pick_studied_pair(correct: np.ndarray, *, prefer: tuple[int, int] = PAPER_PAIR):
    """The paper's pair if the model answers it correctly, else the nearest correct pair
    with the same ones digits (so the ``_6+_9`` lookup story transfers). Returns
    ``(pair, note)``."""
    if correct[prefer[0], prefer[1]]:
        return prefer, f"paper pair {prefer[0]}+{prefer[1]} answered correctly"
    ones = (prefer[0] % 10, prefer[1] % 10)
    pair = pick_pair_by_class(correct, ones=ones, near=prefer)
    if pair is None:
        raise RuntimeError(f"no correct two-digit pair with ones digits {ones}")
    return pair, (
        f"paper pair {prefer[0]}+{prefer[1]} answered wrong; nearest correct "
        f"same-class pair {pair[0]}+{pair[1]}"
    )


@torch.no_grad()
def greedy_answer(model, tokenizer, input_ids: torch.Tensor, n_gen: int = 6) -> tuple[int, str]:
    """Greedy continuation; returns ``(first_token_id, full_text)``."""
    out = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=n_gen,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    cont = out[0, input_ids.shape[1] :]
    return int(cont[0]), tokenizer.decode(cont)


# ---------------------------------------------------------------------------
# §A Attribution graphs (first-digit / ones-digit / free-text prompts)
# ---------------------------------------------------------------------------


def _attach_labels(pg, size_key: str) -> None:
    """Merge HF-repo feature labels + activation examples into the pruned graph's nodes
    (the multilingual/serve convention, so explorer detail panels are populated)."""
    repo_id = get_spec(f"qwen3-{size_key}").transcoder_repo
    by_layer: dict[int, list[int]] = defaultdict(list)
    for nd in pg.nodes:
        if nd.node_type == "feature":
            by_layer[nd.layer].append(nd.feature_idx)
    labels: dict[tuple[int, int], dict] = {}
    for layer, idxs in by_layer.items():
        labs = load_feature_labels(repo_id, layer, idxs)
        exs = load_feature_examples(repo_id, layer, idxs, n_per_quantile=5)
        for fidx in idxs:
            if fidx not in labs and fidx not in exs:
                continue
            d = labs[fidx].to_dict() if fidx in labs else {}
            d["examples"] = exs.get(fidx, [])
            labels[(layer, fidx)] = d
    for nd in pg.nodes:
        if nd.node_type == "feature":
            nd.label = labels.get((nd.layer, nd.feature_idx))


def build_text_graph(
    model,
    tc,
    tokenizer,
    prompt: str,
    *,
    size_key: str = "4b",
    max_feature_nodes: int = 8000,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    out_html=None,
    title: str = "",
    metadata: dict | None = None,
):
    """Build -> prune -> label an attribution graph for *prompt*'s next token.

    Returns ``(pruned_dict, answer_token_id, input_ids)``; writes explorer HTML if
    asked. Raw tokenization (sink prepend), ``n_bos_tokens=1``.
    """
    device = next(model.parameters()).device
    input_ids = tokenize_raw(tokenizer, prompt, device)
    with torch.no_grad():
        answer_id = int(model(input_ids).logits[0, -1].argmax())

    graph = build_attribution_graph(
        model, tc, input_ids, n_bos_tokens=N_BOS, max_feature_nodes=max_feature_nodes
    )
    pruned = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    pg = pruned.graph
    _attach_labels(pg, size_key)

    logit_strs = {
        str(n.token_id): tokenizer.decode(n.token_id) for n in pg.nodes if n.node_type == "logit"
    }
    pruned_dict = graph_to_dict(
        pg,
        influence_scores=pruned.influence_scores,
        prompt=prompt,
        answer_token=tokenizer.decode(answer_id),
        tokens=[tokenizer.decode(t) for t in input_ids[0]],
        logit_token_strs=logit_strs,
        **(metadata or {}),
    )
    if out_html is not None:
        render_graph_explorer_html(pruned_dict, out_html, title=title or prompt[:40])
    return pruned_dict, answer_id, input_ids


def build_calc_graph(
    model,
    tc,
    tokenizer,
    a: int,
    b: int,
    *,
    target: str = "first",
    size_key: str = "4b",
    **kwargs,
):
    """Attribution graph for the studied problem at one of its two answer moments.

    ``target="first"`` analyzes ``calc: a+b=`` -> the sum's first digit;
    ``target="ones"`` teacher-forces the leading digit(s) and analyzes ``...=9`` -> the
    ones digit (digit-by-digit tokenization splits the paper's single "95" circuit).
    """
    if target not in ("first", "ones"):
        raise ValueError(f"target must be 'first' or 'ones', got {target!r}")
    suffix = str(a + b)[:-1] if target == "ones" else ""
    gd, answer_id, input_ids = build_text_graph(
        model,
        tc,
        tokenizer,
        addition_prompt(a, b, suffix),
        size_key=size_key,
        metadata={"pair": [a, b], "target": target},
        **kwargs,
    )
    expected = str(a + b)[-1] if target == "ones" else str(a + b)[0]
    got = tokenizer.decode(answer_id).strip()
    if got != expected:
        raise RuntimeError(
            f"{addition_prompt(a, b, suffix)!r}: greedy answer {got!r} != expected"
            f" {expected!r} — pick a studied pair the model answers correctly"
        )
    return gd, answer_id, input_ids


# ---------------------------------------------------------------------------
# §B Operand grids (the paper's operand plots) + receptive-field classifier
# ---------------------------------------------------------------------------


def _capture_mlp_inputs(model, layers, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    """One forward pass capturing each requested layer's MLP *input* (what the
    transcoders encode — the same signal the graph builder uses)."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_mod, inp, _out):
            captured[layer] = inp[0].detach()

        return hook

    for layer in layers:
        mod = model.get_submodule(f"model.layers.{layer}.mlp")
        handles.append(mod.register_forward_hook(make_hook(layer)))
    try:
        with torch.no_grad():
            model(input_ids, attention_mask=torch.ones_like(input_ids))
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def operand_grids(
    model,
    tc,
    tokenizer,
    features,
    a_vals=None,
    b_vals=None,
    *,
    probe: str = "final",
    batch_size: int = 256,
) -> dict[tuple[int, int], np.ndarray]:
    """The paper's operand plots: each feature's activation over every pair (a, b).

    ``features`` is a list of ``(layer, feature_idx)``. One readout pass per call — the
    cost (~10k forwards) is independent of the feature count. Probes:

    * ``"final"`` — activation at the ``=`` token of ``calc: a+b=`` (the paper's
      operand-plot convention; the first-digit moment);
    * ``"ones"``  — activation at the last token of ``calc: a+b=`` + the sum's leading
      digit(s) (the ones-digit moment; sums < 10 reduce to the ``=`` token);
    * ``"peak"``  — max over all non-sink positions (input-side features living on the
      operand digit tokens).

    Returns ``{(layer, feature_idx): float32 array [len(a_vals), len(b_vals)]}``.
    """
    if probe not in ("final", "ones", "peak"):
        raise ValueError(f"unknown probe {probe!r}")
    a_vals = list(range(100)) if a_vals is None else list(a_vals)
    b_vals = list(range(100)) if b_vals is None else list(b_vals)
    device = next(model.parameters()).device

    feats_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, fidx in features:
        feats_by_layer[int(layer)].append(int(fidx))
    layers = sorted(feats_by_layer)

    grids = {
        (layer, f): np.zeros((len(a_vals), len(b_vals)), dtype=np.float32)
        for layer in layers
        for f in feats_by_layer[layer]
    }
    ai = {a: i for i, a in enumerate(a_vals)}
    bi = {b: j for j, b in enumerate(b_vals)}

    by_len: dict[int, list[tuple[int, int]]] = defaultdict(list)
    ids_cache: dict[tuple[int, int], torch.Tensor] = {}
    for a in a_vals:
        for b in b_vals:
            suffix = str(a + b)[:-1] if probe == "ones" else ""
            ids = tokenize_calc(tokenizer, a, b, device, suffix=suffix)[0]
            ids_cache[(a, b)] = ids
            by_len[ids.shape[0]].append((a, b))

    col_ix = {layer: torch.tensor(feats_by_layer[layer], device=device) for layer in layers}
    for _n, pairs in sorted(by_len.items()):
        for k in range(0, len(pairs), batch_size):
            chunk = pairs[k : k + batch_size]
            ids = torch.stack([ids_cache[p] for p in chunk])
            captured = _capture_mlp_inputs(model, layers, ids)
            for layer in layers:
                x = captured[layer]
                if probe == "peak":
                    acts = tc.transcoders[layer].encode(x)[:, N_BOS:, col_ix[layer]].amax(dim=1)
                else:
                    acts = tc.transcoders[layer].encode(x[:, -1:, :])[:, 0, col_ix[layer]]
                acts = acts.float().cpu().numpy()
                for r, (a, b) in enumerate(chunk):
                    for c, f in enumerate(feats_by_layer[layer]):
                        grids[(layer, f)][ai[a], bi[b]] = acts[r, c]
    return grids


def classify_grid(grid: np.ndarray, *, on_frac: float = 0.25) -> tuple[str, dict]:
    """Name a grid's receptive-field geometry (the paper's operand-plot families).

    Cells above ``on_frac * max`` count as "on". Measures how concentrated the total
    activation mass is on (i) a single (a%10, b%10) lattice cell -> ``lookup``, (ii) a
    single (a+b)%10 anti-diagonal class -> ``mod10-sum``, (iii) a single a%10 / b%10
    class -> ``mod10-a`` / ``mod10-b``, (iv) a contiguous magnitude window of rows /
    columns / anti-diagonals -> ``band-a`` / ``band-b`` / ``sum-band``, (v) a small
    bounding box -> ``region`` / ``point``. Falls through to ``sparse`` / ``mixed``.
    Returns ``(label, stats)``; thresholds are heuristics for seeding and evidence —
    the human review in the explorer is authoritative.
    """
    g = np.nan_to_num(np.asarray(grid, dtype=np.float64), nan=0.0)
    g = np.clip(g, 0.0, None)
    nA, nB = g.shape
    vmax, total = float(g.max()), float(g.sum())
    stats: dict = {"vmax": round(vmax, 4)}
    if vmax <= 0 or total <= 0:
        return "dead", stats

    on = g >= on_frac * vmax
    frac_on = float(on.mean())
    stats["frac_on"] = round(frac_on, 4)

    aa, bb = np.meshgrid(np.arange(nA), np.arange(nB), indexing="ij")
    mass = g / total

    def conc(classes: np.ndarray, n: int) -> tuple[int, float]:
        shares = [float(mass[classes == r].sum()) for r in range(n)]
        r = int(np.argmax(shares))
        return r, shares[r]

    ra, ca = conc(aa % 10, 10)
    rb, cb = conc(bb % 10, 10)
    rs, cs = conc((aa + bb) % 10, 10)
    cell_share = float(mass[(aa % 10 == ra) & (bb % 10 == rb)].sum())
    stats.update(
        mod10_a=[ra, round(ca, 3)],
        mod10_b=[rb, round(cb, 3)],
        mod10_sum=[rs, round(cs, 3)],
        lattice_cell=[ra, rb, round(cell_share, 3)],
    )

    def best_window(profile: np.ndarray, width: int) -> tuple[int, float]:
        if len(profile) <= width:
            return 0, 1.0
        c = np.concatenate([[0.0], np.cumsum(profile)])
        shares = c[width:] - c[:-width]
        s = int(np.argmax(shares))
        return s, float(shares[s] / max(profile.sum(), 1e-12))

    row_prof = mass.sum(axis=1)
    col_prof = mass.sum(axis=0)
    diag_prof = np.array([float(mass[aa + bb == s].sum()) for s in range(nA + nB - 1)])

    # Bands trigger on either the FULL mass (wide/low-precision bands) or the CORE mass
    # (cells >= 0.7*max — a razor band whose broad weak side-structure dilutes the full
    # window). The core path additionally requires the window's core to SPAN the band
    # axis (>= half the grid), so a compact hotspot can't pass as a band.
    core_mask = g >= 0.7 * vmax
    core = np.where(core_mask, mass, 0.0)
    diag_core = np.array([float(core[aa + bb == s].sum()) for s in range(nA + nB - 1)])

    def band_score(profile, core_profile, width, span_of) -> tuple[int, float]:
        start, share = best_window(profile, width)
        if share > 0.6:
            return start, share
        cstart, cshare = best_window(core_profile, width)
        if cshare > 0.7 and span_of(cstart, width) >= nB // 2:
            return cstart, cshare
        return start, 0.0

    def row_span(s, w):
        return int(core_mask[s : s + w].any(axis=0).sum())

    def col_span(s, w):
        return int(core_mask[:, s : s + w].any(axis=1).sum())

    def diag_span(s, w):
        sel = core_mask & (aa + bb >= s) & (aa + bb < s + w)
        return int(np.unique((aa - bb)[sel]).size)

    a0, band_a = band_score(row_prof, core.sum(axis=1), 15, row_span)
    b0, band_b = band_score(col_prof, core.sum(axis=0), 15, col_span)
    s0, band_s = band_score(diag_prof, diag_core, 15, diag_span)
    stats.update(
        band_a=[a0, round(band_a, 3)],
        band_b=[b0, round(band_b, 3)],
        sum_band=[s0, round(band_s, 3)],
    )

    top = np.unravel_index(int(g.argmax()), g.shape)
    top_share = float(mass[top])
    ys, xs = np.where(on)
    bbox = (int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)) if len(ys) else (0, 0)
    bbox_share = (
        float(mass[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].sum()) if len(ys) else 0.0
    )
    stats.update(peak=[int(top[0]), int(top[1])], top_share=round(top_share, 3), bbox=list(bbox))
    # exact-operand cross: a single row AND a single column each carry real mass (the
    # paper's "36"/"59" input-identity features fire when EITHER operand is the value)
    r_star, c_star = int(np.argmax(row_prof)), int(np.argmax(col_prof))
    row_share, col_share = float(row_prof[r_star]), float(col_prof[c_star])
    stats.update(cross=[r_star, c_star, round(row_share, 3), round(col_share, 3)])

    if top_share > 0.5:
        return f"point({top[0]},{top[1]})", stats
    if cell_share >= 0.45:
        return f"lookup(a%10={ra},b%10={rb})", stats
    if min(row_share, col_share) > 0.28 and row_share + col_share > 0.7:
        val = f"{r_star}" if r_star == c_star else f"a={r_star},b={c_star}"
        return f"cross({val})", stats
    if cs > 0.5:
        return f"mod10-sum(r{rs})", stats
    if ca > 0.5:
        return f"mod10-a(r{ra})", stats
    if cb > 0.5:
        return f"mod10-b(r{rb})", stats
    if bbox[0] <= 35 and bbox[1] <= 35 and bbox_share > 0.65:
        return f"region(~{top[0]},~{top[1]})", stats
    if band_s > 0 and band_s >= max(band_a, band_b):
        return f"sum-band(~{s0 + 7})", stats
    if band_a > 0 and band_a >= band_b:
        return f"band-a(~{a0 + 7})", stats
    if band_b > 0:
        return f"band-b(~{b0 + 7})", stats
    if frac_on < 0.02:
        return "sparse", stats
    return "mixed", stats


def on_pair_fraction(grid: np.ndarray, pair: tuple[int, int]) -> float:
    """The studied pair's cell as a fraction of the grid max (0 when the grid is dead)."""
    vmax = float(np.max(grid))
    return float(grid[pair[0], pair[1]] / vmax) if vmax > 0 else 0.0


# ---------------------------------------------------------------------------
# Reviewed supernodes (selection lives in the reviewed file, not notebook code)
# ---------------------------------------------------------------------------


def load_supernodes(path) -> dict[str, dict]:
    """Load the REVIEWED addition supernode file (``build_supernodes.py
    --from-exports``).

    Refuses files that are not ``approved``, contain ``rejected`` members, or repeat a
    ``(layer, feature)`` across supernodes of the same graph (addition supernodes are
    per-graph selections; the same feature MAY recur across graphs — e.g. the lookup
    features re-appearing at the reuse prompt's ones moment). Returns
    ``{name: supernode_dict}``.
    """
    doc = json.loads(Path(path).read_text())
    if not doc.get("approved"):
        raise ValueError(f"{path}: approved=false — review the supernode file first")
    owner: dict[tuple, str] = {}
    out: dict[str, dict] = {}
    for sn in doc["supernodes"]:
        for m in sn["members"]:
            if m.get("review") == "rejected":
                raise ValueError(
                    f"{path}: rejected member L{m['layer']}f{m['feature']} still in"
                    f" {sn['name']!r} members"
                )
            key = (sn.get("graph"), m["layer"], m["feature"])
            if key in owner and owner[key] != sn["name"]:
                raise ValueError(
                    f"{path}: L{key[1]}f{key[2]} in both {owner[key]!r} and {sn['name']!r}"
                    f" on graph {key[0]!r} — per-graph supernodes are disjoint"
                )
            owner[key] = sn["name"]
        out[sn["name"]] = sn
    return out


def resolve_position(sn: dict, member: dict, final_pos: int) -> int:
    """A member's steering position: its own recorded position, falling back to the
    supernode's (``"final"``/None resolve to *final_pos*)."""
    pos = member.get("position")
    if pos is None:
        pos = sn.get("position")
    return final_pos if pos in (None, "final") else int(pos)


def suppress_ivs(sn: dict, *, mult: float, final_pos: int) -> list[FeatureIntervention]:
    """Suppression list: every member steered to ``mult x clean`` (``m = mult - 1``) at
    its recorded position (paper suppressions act at the member's own node)."""
    return [
        FeatureIntervention(
            m["layer"], m["feature"], position=resolve_position(sn, m, final_pos), m=mult - 1.0
        )
        for m in sn["members"]
    ]


def inject_ivs(sn: dict, *, mult: float, positions) -> tuple[list[FeatureIntervention], list]:
    """Donor-injection list: ``value = mult x`` each member's stored (donor-prompt)
    activation, applied at every recipient position in *positions*. Returns
    ``(ivs, used)`` with ``used = [(layer, feature, stored_act)]``."""
    ivs, used = [], []
    for m in sn["members"]:
        act = float(m.get("act") or 0.0)
        if act <= 0:
            continue
        used.append((m["layer"], m["feature"], act))
        ivs.extend(
            FeatureIntervention(m["layer"], m["feature"], position=int(p), value=mult * act)
            for p in positions
        )
    return ivs, used


def choose_end_layer(
    model, tc, input_ids, ivs, token_id: int, *, mode: str = "suppress"
) -> tuple[int, object]:
    """The paper's constrained-patching layer knob: sweep the patch end layer at the
    endpoint strengths and pick the most-suppressing (``mode="suppress"``, for source
    suppressions) or most-promoting (``mode="promote"``, for donor swaps) ``ell``."""
    if mode not in ("suppress", "promote"):
        raise ValueError(f"unknown mode {mode!r}")
    sweep = sweep_patch_end_layer(model, tc, input_ids, ivs, token_id, n_bos_tokens=N_BOS)
    ell = sweep.best_end_layer if mode == "suppress" else sweep.most_promoting_end_layer
    return ell, sweep


# ---------------------------------------------------------------------------
# Intervention reporting (digit distributions, supernode %-readouts)
# ---------------------------------------------------------------------------


@torch.no_grad()
def digit_direct_weights(model, tc, features, tokenizer) -> dict[tuple[int, int], list[float]]:
    """Each feature's DIRECT decoder weight on the ten digit tokens.

    Same math as ``multilingual_helper.direct_logit_effect`` (demeaned unembedding
    projection of the decoder vector through the final RMSNorm weight), batched over
    features and restricted to "0".."9" — the sum-side evidence that e.g. ``sum = _5``
    features write the "5" logit. Returns ``{(layer, feature): [w0..w9]}``.
    """
    device = next(model.parameters()).device
    gamma = model.get_submodule("model.norm").weight.float()
    wu = model.get_submodule("lm_head").weight.float()
    digit_ids = _digit_token_ids(tokenizer)
    feats_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, fidx in features:
        feats_by_layer[int(layer)].append(int(fidx))
    out: dict[tuple[int, int], list[float]] = {}
    for layer, idxs in sorted(feats_by_layer.items()):
        dec = tc.transcoders[layer]._get_decoder_vectors(torch.tensor(idxs, device=device))
        contrib = (gamma * dec.float()) @ wu.T  # (n_feats, vocab)
        demeaned = contrib - contrib.mean(dim=1, keepdim=True)
        for c, fidx in enumerate(idxs):
            out[(layer, fidx)] = [round(float(demeaned[c, t]), 5) for t in digit_ids]
    return out


def digit_distribution(logit_row: torch.Tensor, tokenizer) -> tuple[np.ndarray, float]:
    """P(next token = digit d) for d in 0..9 (full-vocab softmax) and the distribution's
    participation ratio ``1 / sum(p_hat^2)`` over the renormalized digit probabilities —
    the "smear width" (1 = one digit, 10 = uniform)."""
    probs = logit_row.float().softmax(-1)
    p = np.array([float(probs[t]) for t in _digit_token_ids(tokenizer)])
    total = p.sum()
    width = float(1.0 / np.sum((p / total) ** 2)) if total > 0 else float("nan")
    return p, width


def readout_pct(
    res, sn: dict, *, final_pos: int, patch_end_layer: int | None = None, ref: str = "baseline"
) -> dict:
    """Per-supernode %-readout (per-feature ratio first, then mean — a big feature must
    not drown out the others).

    Steered activation / reference x 100 per member, averaged. ``ref="baseline"`` uses
    each member's clean activation on this prompt; ``ref="stored"`` its stored file
    ``act`` (donor groups whose recipient baseline is ~0). Members at layers <= the
    constrained ``patch_end_layer`` are protocol-clamped, not network responses — they
    are excluded and counted as ``n_pinned``. Requires the run to have passed
    ``readout_layers`` covering the supernode's layers.
    """
    per: dict[str, float | str | None] = {}
    ratios: list[float] = []
    skipped = pinned = 0
    for m in sn["members"]:
        layer, fidx = m["layer"], m["feature"]
        if patch_end_layer is not None and layer <= patch_end_layer:
            per[f"L{layer}f{fidx}"] = "pinned"
            pinned += 1
            continue
        if layer not in res.ablated_features:
            raise KeyError(f"layer {layer} missing — pass readout_layers covering {sn['name']!r}")
        base, abl = res.baseline_features[layer], res.ablated_features[layer]
        base = base[0] if base.dim() == 3 else base
        abl = abl[0] if abl.dim() == 3 else abl
        pos = resolve_position(sn, m, final_pos)
        reference = float(base[pos, fidx]) if ref == "baseline" else float(m.get("act") or 0.0)
        if abs(reference) < 1e-6:
            per[f"L{layer}f{fidx}"] = None
            skipped += 1
            continue
        pct = float(abl[pos, fidx]) / reference * 100.0
        per[f"L{layer}f{fidx}"] = round(pct, 1)
        ratios.append(pct)
    mean = round(sum(ratios) / len(ratios), 1) if ratios else None
    return {
        "mean_pct": mean,
        "n_used": len(ratios),
        "n_skipped": skipped,
        "n_pinned": pinned,
        "per_feature": per,
    }


def top_token_probs(logit_row: torch.Tensor, tokenizer, k: int = 6):
    p = logit_row.float().softmax(-1)
    v, i = p.topk(k)
    return [
        (tokenizer.decode([int(t)]).strip(), round(float(pv), 3))
        for pv, t in zip(v, i, strict=True)
    ]


def steer_report(
    model,
    tc,
    tokenizer,
    input_ids,
    ivs,
    *,
    patch_end_layer: int | None,
    readout_sns: dict | None = None,
    final_pos: int | None = None,
    k: int = 6,
) -> dict:
    """Run one constrained intervention and report the answer-position shift.

    Returns baseline/steered top tokens, the digit distribution + smear width
    before/after, and (when *readout_sns* = ``{name: (sn, ref)}`` is given) the
    %-readout of each downstream supernode. ``final_pos`` defaults to the last position.
    """
    final_pos = input_ids.shape[1] - 1 if final_pos is None else final_pos
    readout_sns = readout_sns or {}
    layers = sorted(
        {
            m["layer"]
            for sn, _ref in readout_sns.values()
            for m in sn["members"]
            if patch_end_layer is None or m["layer"] > patch_end_layer
        }
    )
    res = run_feature_intervention(
        model,
        tc,
        input_ids,
        ivs,
        n_bos_tokens=N_BOS,
        patch_end_layer=patch_end_layer,
        readout_layers=layers or None,
    )
    base_row, steer_row = res.baseline_logits[-1], res.ablated_logits[-1]
    p0, w0 = digit_distribution(base_row, tokenizer)
    p1, w1 = digit_distribution(steer_row, tokenizer)
    out = {
        "patch_end_layer": patch_end_layer,
        "top_before": top_token_probs(base_row, tokenizer, k),
        "top_after": top_token_probs(steer_row, tokenizer, k),
        "digits_before": p0.round(4).tolist(),
        "digits_after": p1.round(4).tolist(),
        "width_before": round(w0, 2),
        "width_after": round(w1, 2),
        "readouts": {},
    }
    for name, (sn, ref) in readout_sns.items():
        out["readouts"][name] = readout_pct(
            res, sn, final_pos=final_pos, patch_end_layer=patch_end_layer, ref=ref
        )
    return out


# ---------------------------------------------------------------------------
# §E-§G Reuse in context, introspection, corpus scan
# ---------------------------------------------------------------------------


@torch.no_grad()
def probe_features_on_prompt(
    model, tc, tokenizer, features, prompt: str, *, act_floor: float = 0.05
):
    """Feature activations on an arbitrary raw prompt (sink prepended).

    Returns ``(acts, active_final, input_ids)``: ``acts[(L, f)]`` = full per-position
    activation list; ``active_final`` = ``[(L, f, act)]`` for features above
    *act_floor* at the final position, strongest first.
    """
    device = next(model.parameters()).device
    input_ids = tokenize_raw(tokenizer, prompt, device)
    feats_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, fidx in features:
        feats_by_layer[int(layer)].append(int(fidx))
    captured = _capture_mlp_inputs(model, sorted(feats_by_layer), input_ids)
    acts: dict[tuple[int, int], list[float]] = {}
    active_final = []
    for layer, idxs in sorted(feats_by_layer.items()):
        enc = tc.transcoders[layer].encode(captured[layer])[0, :, idxs].float().cpu().numpy()
        for c, fidx in enumerate(idxs):
            acts[(layer, fidx)] = [round(float(v), 4) for v in enc[:, c]]
            if enc[-1, c] > act_floor:
                active_final.append((layer, fidx, round(float(enc[-1, c]), 4)))
    active_final.sort(key=lambda t: -t[2])
    return acts, active_final, input_ids


INTROSPECTION_QUESTION = "Briefly, how did you get that?"


@torch.no_grad()
def introspection_dialogue(model, tokenizer, a: int, b: int, *, n_gen: int = 160) -> dict:
    """The paper's introspection probe: ask for the answer, then ask *how* (chat format,
    thinking disabled). The paper's model describes the carry algorithm — which is not
    the mechanism the graphs show. Returns the prompt/answer/explanation strings."""
    device = next(model.parameters()).device
    q1 = f"What is {a}+{b}? Answer with just the number."
    messages, _, tkw = prepare_messages(q1, "qwen3", enable_thinking=False)
    ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **tkw
    ).to(device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()

    messages = [
        *messages,
        {"role": "assistant", "content": answer},
        {"role": "user", "content": INTROSPECTION_QUESTION},
    ]
    ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **tkw
    ).to(device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=n_gen,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    explanation = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()
    return {
        "question": q1,
        "answer": answer,
        "how": INTROSPECTION_QUESTION,
        "explanation": explanation,
    }


@torch.no_grad()
def corpus_scan(
    model, tc, tokenizer, features, texts, *, max_tokens: int = 128, act_floor: float = 0.5
):
    """Where do the circuit's features fire in the wild? Scans *texts* and records, per
    feature, every position whose activation clears *act_floor* (token-window snippet
    included). Encode-only (no ``W_dec``), so lazy transcoders are fine."""
    device = next(model.parameters()).device
    feats_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, fidx in features:
        feats_by_layer[int(layer)].append(int(fidx))
    layers = sorted(feats_by_layer)
    hits: dict[tuple[int, int], list[dict]] = {(int(f[0]), int(f[1])): [] for f in features}
    for ti, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        ids = ids[:, :max_tokens].to(device)
        if ids.shape[1] < 4:
            continue
        special = tokenizer.bos_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id
        ids = torch.cat([torch.full((1, 1), special, dtype=ids.dtype, device=device), ids], dim=1)
        captured = _capture_mlp_inputs(model, layers, ids)
        toks = [tokenizer.decode([int(t)]) for t in ids[0]]
        for layer in layers:
            idxs = feats_by_layer[layer]
            enc = tc.transcoders[layer].encode(captured[layer])[0, :, idxs].float().cpu().numpy()
            for c, fidx in enumerate(idxs):
                for pos in np.nonzero(enc[N_BOS:, c] > act_floor)[0] + N_BOS:
                    lo, hi = max(0, pos - 8), min(len(toks), pos + 3)
                    snippet = (
                        "".join(toks[lo:pos]) + "⟦" + toks[pos] + "⟧" + "".join(toks[pos + 1 : hi])
                    )
                    hits[(layer, fidx)].append(
                        {
                            "text": ti,
                            "pos": int(pos),
                            "act": round(float(enc[pos, c]), 3),
                            "snippet": snippet,
                        }
                    )
    for lst in hits.values():
        lst.sort(key=lambda h: -h["act"])
    return hits


# ---------------------------------------------------------------------------
# Plotting (operand grids, accuracy)
# ---------------------------------------------------------------------------


def plot_grid(
    grid: np.ndarray,
    *,
    ax=None,
    title: str = "",
    mark: tuple[int, int] | None = None,
    guide_residue: int | None = None,
):
    """One operand plot: x = b (right operand), y = a (left operand), origin lower,
    magma — the explorer panel's convention. ``mark`` rings a pair; ``guide_residue``
    overlays the (a+b)%10 == r anti-diagonals."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.1, 3.1))
    ax.imshow(grid, origin="lower", cmap="magma", aspect="equal", interpolation="nearest")
    if guide_residue is not None:
        nA, nB = grid.shape
        aa, bb = np.meshgrid(np.arange(nA), np.arange(nB), indexing="ij")
        mask = np.ma.masked_where((aa + bb) % 10 != guide_residue % 10, np.ones_like(grid))
        ax.imshow(mask, origin="lower", cmap="cool", alpha=0.25, aspect="equal")
    if mark is not None:
        ax.scatter([mark[1]], [mark[0]], s=60, facecolors="none", edgecolors="#00e676", lw=1.4)
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(title, fontsize=8)
    return ax


def plot_grid_panels(items, *, ncols: int = 6, mark=None, panel: float = 2.2):
    """Figure of operand plots: *items* = ``[(grid, title), ...]`` (the paper's Fig-A1
    style array; also the per-supernode evidence figure)."""
    n = len(items)
    ncols = min(ncols, max(n, 1))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows + 0.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, (grid, title) in zip(axes, items, strict=False):
        plot_grid(grid, ax=ax, title=title, mark=mark)
        ax.set_xticks([0, 50, 99])
        ax.set_yticks([0, 50, 99])
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_accuracy(correct: np.ndarray, *, ax=None, title: str = "greedy accuracy on a+b"):
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(correct.astype(float), origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"{title} ({correct.mean() * 100:.1f}%)", fontsize=9)
    return ax
