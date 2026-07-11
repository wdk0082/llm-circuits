"""Addition-dive helpers for the paper-exact reproduction on Qwen3 (biology.html §Addition
+ the methods-paper addition supplement).

Single helper module for ``addition.ipynb`` — merges the former ``helper.py`` (prompts,
accuracy/operand grids, graph building, heatmap taxonomy), ``addition_paper.py``
(paper-protocol machinery: raw-prompt graphs, position supernodes, steer-and-report,
introspection, corpus scan) and the non-CLI logic of the retired
``examples/paper_addition_*.py`` scripts (studied-pair selection, panel figures,
polymer-reuse probing).

Deliberately kept *out* of the core ``llm_circuits`` package: everything here is
addition-task-specific (prompt formatting, the operand-grid feature sweep, the
mod-10/diagonal heatmap classifier) and doesn't generalise.

Conventions: paper steering multiples are MULTIPLICATIVE (``M_paper``); ours are
additive-delta (``m = M_paper - 1``). Paper -2x suppression == m=-3; sign-flip -1x ==
m=-2; ablation 0x == m=-1.
"""

from __future__ import annotations

import os
from collections import defaultdict

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

# Two prompt styles:
#   "chat" — natural-language instruct format (bare "a+b=" makes the instruct model explain
#            instead of answering); matches examples/addition_circuit.py.
#   "calc" — the PAPER's exact raw-completion format ``calc: a+b=`` (biology.html uses
#            10,000 prompts of this form, feature activity read on the ``=`` token). Raw
#            completion, no chat template; a special token is prepended as the attention
#            sink (circuit-tracer's convention) so n_bos_tokens=1 still holds.
PROMPT_TEMPLATE = "What is {a}+{b}? Answer with just the number."
CALC_TEMPLATE = "calc: {a}+{b}="
N_BOS = 1  # one BOS-like token (chat template's first token, or our raw-prompt prepend)

# The paper's exact auxiliary prompts (Fig A4-A6), raw completions.
POLYMER_PROMPT = "K. Whang, etc. Polymer, 36, 837, 1"  # paper: completes "995"
INTERMEDIATE_PROMPT = "assert (4 + 5) * 3 =="  # paper: completes "27"

DIGIT_TOKENS = [str(d) for d in range(10)]


def addition_prompt(a: int, b: int, style: str = "chat") -> str:
    tmpl = CALC_TEMPLATE if style == "calc" else PROMPT_TEMPLATE
    return tmpl.format(a=a, b=b)


def tokenize(tokenizer, prompt: str, device) -> torch.Tensor:
    """Chat-template tokenisation identical to the graph builder (so positions line up)."""
    messages, _, tkw = prepare_messages(prompt, "qwen3", enable_thinking=False)
    return tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **tkw
    ).to(device)


def tokenize_raw(tokenizer, prompt: str, device) -> torch.Tensor:
    """Raw-completion tokenisation with a prepended special token (the attention sink).

    Mirrors circuit-tracer's ``ensure_tokenized``: transcoders can't reconstruct position 0,
    so raw prompts get a special token (eos/pad — Qwen3 has no BOS) prepended and every
    downstream consumer keeps ``n_bos_tokens=1``.
    """
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    special = tokenizer.bos_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id
    ids = torch.cat([torch.tensor([[special]], dtype=ids.dtype), ids], dim=1)
    return ids.to(device)


def tokenize_addition(tokenizer, a: int, b: int, device, style: str = "chat") -> torch.Tensor:
    prompt = addition_prompt(a, b, style)
    return (
        tokenize_raw(tokenizer, prompt, device)
        if style == "calc"
        else tokenize(tokenizer, prompt, device)
    )


# ---------------------------------------------------------------------------
# Accuracy sweep + studied-pair selection
# ---------------------------------------------------------------------------


def accuracy_grid(
    model, tokenizer, a_vals, b_vals, batch_size: int = 256, n_gen: int = 4, *, style: str = "chat"
):
    """Greedy multi-token generation for every (a, b); returns (correct, predicted grids).

    Qwen3 tokenises numbers **digit-by-digit** ("95" -> ['9','5']), so we greedily generate
    ``n_gen`` tokens and compare the leading integer of the decoded continuation to
    ``str(a+b)``.  ``correct[i, j]`` is True iff they match.  ``style="calc"`` uses the
    paper's raw ``calc: a+b=`` format.
    """
    import re

    device = next(model.parameters()).device
    correct = np.zeros((len(a_vals), len(b_vals)), dtype=bool)
    predicted = np.empty((len(a_vals), len(b_vals)), dtype=object)
    ai = {a: i for i, a in enumerate(a_vals)}
    bi = {b: j for j, b in enumerate(b_vals)}

    bylen: dict[int, list] = defaultdict(list)
    for a in a_vals:
        for b in b_vals:
            ids = tokenize_addition(tokenizer, a, b, device, style)
            bylen[ids.shape[1]].append((a, b, ids))

    with torch.no_grad():
        for seqlen, items in bylen.items():
            for s in range(0, len(items), batch_size):
                chunk = items[s : s + batch_size]
                batch = torch.cat([ids for _, _, ids in chunk], dim=0)
                # KV-cached greedy generation (uniform length within a group -> no padding).
                out = model.generate(
                    batch,
                    attention_mask=torch.ones_like(batch),
                    max_new_tokens=n_gen,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                for k, (a, b, _) in enumerate(chunk):
                    cont = tokenizer.decode(out[k, seqlen:]).strip()
                    m = re.match(r"\d+", cont)
                    num = m.group() if m else ""
                    predicted[ai[a], bi[b]] = cont
                    correct[ai[a], bi[b]] = num == str(a + b)
    return correct, predicted


def pick_studied_pair(a0: int, b0: int, correct: np.ndarray) -> tuple[int, int, bool]:
    """The paper studies a prompt the model answers CORRECTLY (Haiku: 36+59 -> 95).

    If the model answers (a0, b0) correctly, keep it.  Otherwise pick the nearest
    two-digit pair with the SAME digit classes — a % 10, b % 10 and a + b all equal to
    the requested pair's (preserving the paper's ``_6 + _9 -> sum = _95`` lookup/sum
    structure) — that the model does answer correctly.
    """
    if correct[a0, b0]:
        return a0, b0, False
    cands = [
        (a, b)
        for a in range(10, 100)
        for b in range(10, 100)
        if a % 10 == a0 % 10 and b % 10 == b0 % 10 and a + b == a0 + b0 and correct[a, b]
    ]
    if not cands:
        raise RuntimeError(
            f"model answers {a0}+{b0} incorrectly and no correct same-class pair exists"
        )
    a, b = min(cands, key=lambda ab: abs(ab[0] - a0) + abs(ab[1] - b0))
    return a, b, True


# ---------------------------------------------------------------------------
# Operand-grid feature activations (the headline visualisation)
# ---------------------------------------------------------------------------


def feature_grids(
    model,
    tc,
    features,
    a_vals,
    b_vals,
    tokenizer,
    batch_size: int = 256,
    *,
    probe: str = "first",
    style: str = "chat",
):
    """Activation of each feature over the full (a, b) grid.

    ``features`` is a list of ``(layer, feature_idx)``; ``probe`` picks **which position**
    the feature is measured at:

    * ``"first"`` — plain prompt; **peak activation over the non-BOS positions** (captures
      the feature whether it fires on an operand or on the predict-first-digit position).
      Use for the **magnitude** circuit and for **input-token** features (which are silent
      at the final position).
    * ``"last"`` — plain prompt, final position (for ``style="calc"`` that is the paper's
      ``=`` token — the paper's answer-feature probe).
    * ``"ones"`` — **teacher-force** all answer digits but the last (``prompt + '9'`` for
      ``a+b=95``) and read the feature at the **final position** — exactly the
      predict-ones-digit position the ``target="ones"`` graph attributes from.  Measuring
      a ones-digit *output* feature on the plain prompt instead would sample it at the
      predict-first-digit position, spuriously making it look like a magnitude diagonal.

    Both use the same ``transcoder.encode(mlp_input)`` the graph builder uses.  Returns
    ``{(layer, feature_idx): np.ndarray[len(a_vals), len(b_vals)]}``.
    """
    device = next(model.parameters()).device
    layers = sorted({L for L, _ in features})
    ai = {a: i for i, a in enumerate(a_vals)}
    bi = {b: j for j, b in enumerate(b_vals)}
    grids = {
        (L, f): np.full((len(a_vals), len(b_vals)), np.nan, dtype=np.float32) for L, f in features
    }

    captured: dict[int, torch.Tensor] = {}

    def mk_hook(layer: int):
        def hook(_m, inp, _o):
            captured[layer] = inp[0]  # MLP input == transcoder input (skip_connection=False)

        return hook

    handles = [
        model.get_submodule(f"model.layers.{L}.mlp").register_forward_hook(mk_hook(L))
        for L in layers
    ]

    bylen: dict[int, list] = defaultdict(list)
    for a in a_vals:
        for b in b_vals:
            ids = tokenize_addition(tokenizer, a, b, device, style)
            if probe == "ones":
                ans = tokenizer(
                    str(a + b), add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(device)
                ids = torch.cat([ids, ans[:, :-1]], dim=1)  # teacher-force the leading digit(s)
            bylen[ids.shape[1]].append((a, b, ids))

    try:
        with torch.no_grad():
            for items in bylen.values():
                for s in range(0, len(items), batch_size):
                    chunk = items[s : s + batch_size]
                    model(torch.cat([ids for _, _, ids in chunk], dim=0))
                    for L in layers:
                        feats = tc.transcoders[L].encode(captured[L])  # (B, seq, d_t)
                        if probe in ("ones", "last"):
                            # final position: predict-ones (teacher-forced) or the prompt's
                            # last token — for style="calc" that is the paper's "=" token.
                            sel = feats[:, -1, :].float().cpu().numpy()
                        else:
                            sel = feats[:, N_BOS:, :].amax(dim=1).float().cpu().numpy()
                        for L2, f in features:
                            if L2 != L:
                                continue
                            col = sel[:, f]
                            for k, (a, b, _) in enumerate(chunk):
                                grids[(L, f)][ai[a], bi[b]] = col[k]
    finally:
        for h in handles:
            h.remove()
    return grids


# ---------------------------------------------------------------------------
# Attribution graphs (chat/calc addition prompts + generic raw prompts)
# ---------------------------------------------------------------------------


def _attach_labels_and_dump(
    pg, pruned, tokenizer, *, size_key: str, prompt: str, answer_id: int, input_ids, out_html, title
):
    """Shared tail of the graph builders: labels -> dict -> optional explorer HTML."""
    repo_id = get_spec(f"qwen3-{size_key}").transcoder_repo
    by_layer: dict[int, list[int]] = defaultdict(list)
    for nd in pg.nodes:
        if nd.node_type == "feature":
            by_layer[nd.layer].append(nd.feature_idx)
    labels: dict[tuple[int, int], dict] = {}
    for layer, idxs in by_layer.items():
        labs = load_feature_labels(repo_id, layer, idxs)
        # Activation examples ship in the same label blobs; merging them here (the serve
        # engine's convention) lets the explorer's detail panel show the quantile-grouped
        # max-activating snippets instead of "no activation examples".
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
    )
    if out_html is not None:
        render_graph_explorer_html(pruned_dict, out_html, title=title)
    return pruned_dict


def build_addition_graph(
    model,
    tc,
    tokenizer,
    a: int,
    b: int,
    *,
    target: str = "first",
    style: str = "chat",
    size_key: str = "4b",
    max_feature_nodes: int = 8000,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    out_html=None,
):
    """Build → prune → label an attribution graph for ``a+b``.

    ``target`` picks which answer digit's circuit to study (Qwen3 emits digits
    left-to-right, one token each):

    * ``"first"`` — the next token after the prompt = the **leading/magnitude** digit.
    * ``"ones"`` — **teacher-force** all answer digits except the last, so the next token
      is the **ones digit** (the paper's lookup-table / mod-10 target).

    Uses the fast activation cap and circuit-tracer's prune thresholds.  Returns
    ``(pruned_dict, answer_token_id, input_ids)``; writes the interactive explorer HTML to
    ``out_html`` if given.
    """
    device = next(model.parameters()).device
    prompt_ids = tokenize_addition(tokenizer, a, b, device, style)
    if target == "ones":
        ans_ids = tokenizer(str(a + b), add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        input_ids = torch.cat([prompt_ids, ans_ids[:, :-1]], dim=1)  # teacher-force the prefix
        answer_id = int(ans_ids[0, -1])  # the ground-truth ones-digit token
    else:
        input_ids = prompt_ids
        with torch.no_grad():
            answer_id = int(model(input_ids).logits[0, -1].argmax())

    graph = build_attribution_graph(
        model, tc, input_ids, n_bos_tokens=N_BOS, max_feature_nodes=max_feature_nodes
    )
    pruned = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    pruned_dict = _attach_labels_and_dump(
        pruned.graph,
        pruned,
        tokenizer,
        size_key=size_key,
        prompt=addition_prompt(a, b, style),
        answer_id=answer_id,
        input_ids=input_ids,
        out_html=out_html,
        title=f"{a}+{b}={a + b}",
    )
    return pruned_dict, answer_id, input_ids


def build_graph_raw(
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
):
    """Build -> prune -> label an attribution graph for a RAW (non-chat) prompt."""
    device = next(model.parameters()).device
    input_ids = tokenize_raw(tokenizer, prompt, device)
    with torch.no_grad():
        answer_id = int(model(input_ids).logits[0, -1].argmax())

    graph = build_attribution_graph(
        model, tc, input_ids, n_bos_tokens=N_BOS, max_feature_nodes=max_feature_nodes
    )
    pruned = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    pruned_dict = _attach_labels_and_dump(
        pruned.graph,
        pruned,
        tokenizer,
        size_key=size_key,
        prompt=prompt,
        answer_id=answer_id,
        input_ids=input_ids,
        out_html=out_html,
        title=title or prompt[:40],
    )
    return pruned_dict, answer_id, input_ids


# ---------------------------------------------------------------------------
# Picking features to visualise + classifying their operand-grid structure
# ---------------------------------------------------------------------------

_NUM_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fifteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "first",
    "second",
    "third",
    "fifth",
    "ninth",
}


def is_number_feature(node) -> bool:
    """True if the feature's top output logits look numeric (a digit or a number word).

    Filters out early-layer 'junk' features whose labels are unrelated tokens.
    """
    import re

    lab = node.get("label") or {}
    for t in (lab.get("top_logits") or [])[:5]:
        s = str(t).strip().lower().lstrip("-_ ")
        if re.search(r"\d", s) or any(w == s or w in s.split() for w in _NUM_WORDS):
            return True
    return False


def answer_features(pruned_dict, top_k: int = 12, *, numeric_only: bool = False):
    """Top-``top_k`` feature nodes by influence (the answer-driving supernode).

    ``pruned_dict`` is ``graph_to_dict(prune_graph(...).graph, influence_scores=...)``.
    With ``numeric_only`` keep only number features (:func:`is_number_feature`).
    Returns a list of dicts with layer / feature_idx / position / influence / label.
    """
    feats = [n for n in pruned_dict["nodes"] if n["node_type"] == "feature"]
    if numeric_only:
        feats = [f for f in feats if is_number_feature(f)]
    feats.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    return feats[:top_k]


def digit_token_positions(tokenizer, input_ids, a: int, b: int) -> dict[str, list[int]]:
    """Positions of the operand digit tokens in a ``calc: a+b=`` prompt.

    Qwen3 tokenises digits one per token, so ``36+59`` -> tokens 3,6,+,5,9.  Returns
    ``{"a_digits": [...], "b_digits": [...], "eq": [pos_of_=], "plus": [pos_of_+]}``
    (positions past N_BOS).  ``plus``/``eq`` are the operator positions where the paper's
    add-function features live.

    Digit collection STOPS at the ``=`` token: on teacher-forced ones prompts the
    sequence continues with forced ANSWER digits, and without the stop the trailing
    ``len(b_str)`` slice kept the forced first answer digit and dropped b's tens digit
    — so ``suppress_9`` steered the predict-ones position instead of the operand's
    ones digit (DEVLOG_EXTRA §1.1; found by the 2026-07-10 pre-report scan).
    """
    toks = [tokenizer.decode([int(t)]) for t in input_ids[0]]
    a_str, b_str = str(a), str(b)
    a_pos: list[int] = []
    b_pos: list[int] = []
    eq_pos: list[int] = []
    plus_pos: list[int] = []
    plus_seen = False
    eq_seen = False
    for i, t in enumerate(toks):
        if i < N_BOS:
            continue
        s = t.strip()
        if s == "+":
            plus_seen = True
            plus_pos.append(i)
        elif s == "=":
            eq_seen = True
            eq_pos.append(i)
        elif s.isdigit() and not eq_seen:
            (b_pos if plus_seen else a_pos).append(i)
    # keep only the trailing len(a_str)/len(b_str) digit tokens (guards against digits
    # appearing earlier in the prompt, e.g. in a preamble).
    return {
        "a_digits": a_pos[-len(a_str) :],
        "b_digits": b_pos[-len(b_str) :],
        "eq": eq_pos,
        "plus": plus_pos,
    }


def position_features(pruned_dict, positions, top_n: int = 12, *, max_layer: int | None = None):
    """Influence-ranked feature nodes at any of ``positions`` -> ``[(layer, idx, act)]``.

    ``max_layer`` keeps only layers ``< max_layer`` — the paper's INPUT supernodes
    (``_6`` / ``_9`` / ``~magnitude``) are detokenization-level features at the bottom of
    the graph; without the cap, late-layer aggregation features at the operand positions
    dominate the influence ranking and crowd them out.
    """
    pos = set(positions)
    feats = [
        n
        for n in pruned_dict["nodes"]
        if n["node_type"] == "feature"
        and n["position"] in pos
        and (max_layer is None or n["layer"] < max_layer)
    ]
    feats.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    return [(n["layer"], n["feature_idx"], float(n.get("activation", 0.0))) for n in feats[:top_n]]


def select_by_grid_label(feats, grids, a_vals, b_vals, predicate) -> list[tuple[int, int, float]]:
    """Filter ``[(layer, idx, act)]`` to those whose operand-grid report satisfies
    ``predicate(report)`` — e.g. input ``_6`` features: ``r['label'].startswith('mod10-a')
    and r['a_top'] == 6``."""
    out = []
    for L, i, act in feats:
        rep = periodicity_report(grids[(L, i)], a_vals, b_vals)
        if predicate(rep):
            out.append((L, i, act))
    return out


def select_from_reports(reports: dict[str, dict], predicate) -> list[tuple[int, int]]:
    """Filter a ``{"L{L}f{i}": periodicity_report}`` dict by ``predicate(report)``.

    Returns ``[(layer, idx)]`` — the taxonomy-driven supernode selection used for the
    paper's suppression experiments.
    """
    return [
        tuple(int(x) for x in key[1:].split("f")) for key, rep in reports.items() if predicate(rep)
    ]


def periodicity_report(grid, a_vals, b_vals) -> dict:
    """Quantify the mod-10 / diagonal structure of an operand-grid heatmap.

    Returns metrics that distinguish the paper's feature families:

    * ``sum_ac10`` — autocorrelation of the sum-profile ``P(a+b)`` at lag 10.  High (>~0.4)
      ⇒ the feature fires on **periodic diagonals** (``a+b`` repeating every 10) — the mod-10
      *lookup* signature.  Near 0 ⇒ a single diagonal (magnitude).
    * ``s_conc`` / ``a_conc`` / ``b_conc`` — peak-to-mean ratio of activation grouped by
      ``(a+b)%10`` / ``a%10`` / ``b%10``.  >~1.5 ⇒ concentrated on one residue (ends-in-d).
    * ``s_std`` — spread of ``a+b`` over strongly-active cells (small ⇒ one diagonal).
    * ``a_std`` / ``b_std`` (+ ``a_mean`` / ``b_mean``) — spread/center of a / b over the
      **bright core** (> 0.7·max; the other stats use the 0.5 threshold).  A clean
      one-operand BAND (the paper's magnitude-input features, ``~30`` / ``~59``) is
      narrow in its own operand and broad in the other: core std ≈ 1-6 vs ≈ 20-29 for
      stripes/lattices/uniform.  The stricter core matters because real band features
      carry a WEAK cross-arm in the other operand that sits above 0.5·max (measured on
      qwen3-4b: ``L4f148151``'s arm inflates b-at-0.5 to std ≈ 21 but exits the 0.7
      core), while an exact-value cross (a=46 OR b=46, both arms equally bright) stays
      wide in both and is correctly excluded — the paper's magnitude intervention
      targets the ``~30``/``~59`` bands, not the exact-value features.
    * ``label`` — derived family: ``lookup(a%10=M,b%10=N)`` (jointly residue-selective
      points — the paper's lookup-table signature) / ``mod10-sum(rN)`` / ``band-a(~M)`` /
      ``band-b(~M)`` (one-operand magnitude bands) / ``mod10-a(rN)`` / ``mod10-b(rN)`` /
      ``magnitude-diag`` / ``sparse`` (fires in <1% of cells) / ``mixed`` / ``inactive``.
      Bands are checked BEFORE the single-operand mod-10 stripes: a near-value band
      concentrates on ~5 residues and can hair-trigger the stripe test (measured:
      b_conc 1.52 on a razor b≈49 band), while a true periodic stripe can never have a
      small core std — the two signatures are disjoint under the core statistic.
    """
    g = np.nan_to_num(np.asarray(grid, dtype=float), nan=0.0)
    A = np.repeat(np.asarray(a_vals, int)[:, None], len(b_vals), axis=1)
    B = np.repeat(np.asarray(b_vals, int)[None, :], len(a_vals), axis=0)
    S = A + B
    gmax = float(g.max())
    if gmax <= 0:
        return {"label": "inactive", "max": 0.0}
    gn = g / gmax

    def residue(M):
        means = np.array(
            [gn[r == (M % 10)].mean() if np.any(r == (M % 10)) else 0.0 for r in range(10)]
        )
        return int(means.argmax()), float(means.max() / (means.mean() + 1e-9))

    a_top, a_conc = residue(A)
    b_top, b_conc = residue(B)
    s_top, s_conc = residue(S)

    smin, smax = int(S.min()), int(S.max())
    P = np.array([gn[s == S].mean() if np.any(s == S) else 0.0 for s in range(smin, smax + 1)])
    Pc = P - P.mean()
    denom = float((Pc * Pc).sum()) or 1.0
    sum_ac10 = float((Pc[:-10] * Pc[10:]).sum() / denom) if len(Pc) > 10 else 0.0

    on = gn > 0.5
    s_std = float(S[on].std()) if on.any() else 1e9
    core = gn > 0.7  # bright core: weak band cross-arms sit above 0.5·max but below this
    a_std = float(A[core].std()) if core.any() else 1e9
    b_std = float(B[core].std()) if core.any() else 1e9
    a_mean = float(A[core].mean()) if core.any() else float("nan")
    b_mean = float(B[core].mean()) if core.any() else float("nan")
    frac_on = float(on.mean())

    rep = dict(
        max=gmax,
        a_top=a_top,
        a_conc=a_conc,
        b_top=b_top,
        b_conc=b_conc,
        s_top=s_top,
        s_conc=s_conc,
        sum_ac10=sum_ac10,
        s_std=s_std,
        a_std=a_std,
        b_std=b_std,
        a_mean=a_mean,
        b_mean=b_mean,
        frac_on=frac_on,
    )
    if frac_on <= 0.01:
        rep["label"] = "sparse"
    elif a_conc > 1.5 and b_conc > 1.5:
        # Jointly selective for BOTH operands' residues -> a repeating grid of points:
        # the paper's LOOKUP-TABLE signature (e.g. "_6 + _9").
        rep["label"] = f"lookup(a%10={a_top},b%10={b_top})"
    elif sum_ac10 > 0.4 and s_conc > 1.3:
        rep["label"] = f"mod10-sum(r{s_top})"
    elif a_std < 8 and b_std > 15:
        # Narrow core in a, broad in b: a one-operand magnitude band (paper's "~30").
        # Checked before the mod-10 stripes — a near-value band concentrates on ~5
        # residues (can trip a/b_conc), but no periodic stripe has a small core std.
        rep["label"] = f"band-a(~{round(a_mean)})"
    elif b_std < 8 and a_std > 15:
        rep["label"] = f"band-b(~{round(b_mean)})"
    elif a_conc > 1.5:
        rep["label"] = f"mod10-a(r{a_top})"
    elif b_conc > 1.5:
        rep["label"] = f"mod10-b(r{b_top})"
    elif s_std < 8:
        rep["label"] = "magnitude-diag"
    else:
        rep["label"] = "mixed"
    return rep


def classify_grid(grid, a_vals, b_vals, **_):
    """Back-compat wrapper: returns ``(label, full_report)`` from :func:`periodicity_report`."""
    rep = periodicity_report(grid, a_vals, b_vals)
    return rep["label"], rep


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_grid(
    grid, a_vals, b_vals, *, title="", ax=None, cmap="magma", mark=None, overlay_sum_residue=None
):
    """Operand-grid heatmap: x = right operand b, y = left operand a.

    ``overlay_sum_residue=r`` draws the periodic anti-diagonals ``a+b ≡ r (mod 10)`` so a
    mod-10 (lookup) feature is visually obvious — its bright cells should sit on them.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(4.2, 3.6))
    amin, amax, bmin, bmax = min(a_vals), max(a_vals), min(b_vals), max(b_vals)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, extent=[bmin, bmax, amin, amax])
    if overlay_sum_residue is not None:
        for s in range(amin + bmin, amax + bmax + 1):
            if s % 10 == overlay_sum_residue % 10:
                b0, b1 = max(bmin, s - amax), min(bmax, s - amin)
                if b0 <= b1:
                    ax.plot([b0, b1], [s - b0, s - b1], color="cyan", lw=0.5, alpha=0.45)
    ax.set_xlabel("right operand  b")
    ax.set_ylabel("left operand  a")
    ax.set_title(title, fontsize=9)
    if mark is not None:
        ax.scatter([mark[1]], [mark[0]], s=40, facecolors="none", edgecolors="lime", linewidths=1.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if own:
        plt.tight_layout()
    return ax


def plot_grid_panels(feats, grids, a_vals, b_vals, *, mark, title, path=None, cols: int = 4):
    """Paper-Fig-A1-style panel of operand grids for ``feats = [(layer, idx), ...]``.

    Each panel is titled with the feature id and its :func:`periodicity_report` label.
    Saves to ``path`` if given; returns the figure (display it inline in a notebook).
    """
    rows = (len(feats) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.1 * rows))
    for ax, (L, i) in zip(np.ravel(axes), feats, strict=False):
        rep = periodicity_report(grids[(L, i)], a_vals, b_vals)
        plot_grid(
            grids[(L, i)], a_vals, b_vals, ax=ax, mark=mark, title=f"L{L} f{i}\n{rep['label']}"
        )
    for ax in np.ravel(axes)[len(feats) :]:
        ax.axis("off")
    fig.suptitle(title, y=1.005)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def feature_label(node) -> str:
    """Short human label for a feature node (top output logits, if available)."""
    lab = node.get("label") or {}
    tops = lab.get("top_logits") or []
    base = f"L{node['layer']} f{node['feature_idx']}"
    return f"{base} ({'/'.join(map(str, tops[:3]))})" if tops else base


# ---------------------------------------------------------------------------
# Interventions with feature readout + smear metrics
# ---------------------------------------------------------------------------


def steer_interventions(feats, *, m: float, position: int | None = None):
    """``[(layer, idx, act)]`` -> the :class:`FeatureIntervention` list steering each
    feature by ``m`` at ``position`` (``None`` = every non-BOS position, per
    ``FeatureIntervention`` semantics — how the magnitude supernodes are steered)."""
    return [FeatureIntervention(L, i, position=position, m=m) for (L, i, _) in feats]


def choose_patch_end_layer(
    model, tc, input_ids, interventions, token_id: int, *, mode: str = "suppress"
):
    """The paper's intervention-layer recipe: sweep the constrained-patching end layer
    ``ell`` over ``[l_max, n_layers-1]`` and pick the most effective one on the
    ``token_id`` metric — ``mode="suppress"`` = largest logit suppression (input/feature
    suppressions), ``mode="promote"`` = largest probability (donor injections / swaps).

    Returns ``(ell, sweep)`` where ``sweep`` is the full
    :class:`~llm_circuits.circuits.interventions.LayerSweepResult` curve.
    """
    sweep = sweep_patch_end_layer(model, tc, input_ids, interventions, token_id, n_bos_tokens=N_BOS)
    ell = sweep.best_end_layer if mode == "suppress" else sweep.most_promoting_end_layer
    return ell, sweep


def steer_and_report(
    model,
    tc,
    input_ids,
    feats,
    tokenizer,
    *,
    m: float,
    position: int | None = None,
    readout: list[tuple[int, int, int]] | None = None,
    patch_end_layer: int | None = None,
):
    """Steer ``feats`` (``[(layer, idx, act)]``) by ``m`` at ``position`` (``None`` =
    every non-BOS position — how the magnitude supernodes are steered).

    ``patch_end_layer`` selects the paper's **constrained patching** (activations up to
    layer ``ell`` clamped at their perturbed values, real model after ``ell``); ``None`` =
    fully-propagating clean-anchored deltas (the no-pinning robustness variant).
    Readout entries at layers ``<= patch_end_layer`` are pinned by the protocol and
    reported as ``pinned: True`` (their %-of-baseline is not meaningful).

    Returns dict with top tokens (baseline/steered), the 0-9 digit distribution + smear
    metrics for both, and each ``readout`` feature's activation as % of baseline
    (``readout`` entries are ``(layer, feature_idx, position)``).
    """
    ivs = steer_interventions(feats, m=m, position=position)
    readout_layers = sorted({L for (L, _, _) in readout or []})
    res = run_feature_intervention(
        model,
        tc,
        input_ids,
        ivs,
        n_bos_tokens=N_BOS,
        readout_layers=readout_layers,
        patch_end_layer=patch_end_layer,
    )
    row_b, row_s = res.baseline_logits[-1], res.ablated_logits[-1]
    report = {
        "m": m,
        "n_feats": len(feats),
        "patch_end_layer": patch_end_layer,
        "baseline_top": top_tokens(row_b, tokenizer),
        "steered_top": top_tokens(row_s, tokenizer),
        "baseline_digits": digit_distribution(row_b, tokenizer),
        "steered_digits": digit_distribution(row_s, tokenizer),
    }
    if readout:
        pct = {}
        for L, i, p in readout:
            base = res.baseline_features.get(L)
            after = res.ablated_features.get(L)
            if base is None or after is None:
                continue
            base2 = base[0] if base.dim() == 3 else base
            after2 = after[0] if after.dim() == 3 else after
            b0 = float(base2[p, i])
            a0 = float(after2[p, i])
            entry = {
                "baseline": b0,
                "steered": a0,
                "pct_of_baseline": (a0 / b0 * 100.0) if abs(b0) > 1e-6 else None,
            }
            if patch_end_layer is not None and patch_end_layer >= L:
                entry["pinned"] = True  # clamped by the protocol; % not meaningful
            pct[f"L{L}f{i}@p{p}"] = entry
        report["readout_pct"] = pct
    return report


def top_tokens(row, tokenizer, k: int = 6):
    p = row.float().softmax(-1)
    v, i = p.topk(k)
    return [
        (tokenizer.decode([int(t)]).strip(), round(float(pv), 4))
        for pv, t in zip(v, i, strict=True)
    ]


def digit_distribution(row, tokenizer) -> dict:
    """Probability over the digit tokens '0'-'9' + smear metrics.

    ``width`` is the participation ratio ``1 / sum(p_hat^2)`` over the renormalised digit
    distribution — ~1 for a confident single digit, ~10 for a uniform smear.  The paper's
    "smears the result out over a range of 5" corresponds to width ~5.
    """
    probs = row.float().softmax(-1)
    ids = [tokenizer(d, add_special_tokens=False).input_ids[0] for d in DIGIT_TOKENS]
    p = np.array([float(probs[i]) for i in ids])
    total = p.sum()
    p_hat = p / total if total > 0 else p
    width = float(1.0 / (p_hat**2).sum()) if total > 0 else 0.0
    entropy = float(-(p_hat[p_hat > 0] * np.log2(p_hat[p_hat > 0])).sum()) if total > 0 else 0.0
    return {
        "digit_probs": {d: round(float(v), 4) for d, v in zip(DIGIT_TOKENS, p, strict=True)},
        "digit_mass": round(float(total), 4),
        "width": round(width, 3),
        "entropy_bits": round(entropy, 3),
    }


# ---------------------------------------------------------------------------
# Direct output-weight screen (the paper's computed-9 signature)
# ---------------------------------------------------------------------------


@torch.no_grad()
def direct_token_weights(model, tc, feats, token_id: int) -> dict[tuple[int, int], float]:
    """Demeaned direct unembedding weight of each feature's decoder on ``token_id``.

    Mirrors ``multilingual_helper.direct_logit_effect`` — ``(g * W_dec[feature]) .
    (W_U[token] - mean_t W_U)``, the decoder written through the final RMSNorm weight
    ``g`` and read against the **vocab-demeaned** unembedding row — but batched per
    layer via ``_get_decoder_vectors`` so a whole graph's feature nodes screen in one
    pass.  The paper's "computed 9, intermediate step" feature is defined by its
    strongest *negative* weight on "9" (a suppressive output effect that a top-logit
    label scan cannot find).  Returns ``{(layer, feature_idx): weight}``.
    """
    device = next(model.parameters()).device
    gamma = model.get_submodule("model.norm").weight.float()
    wu = model.get_submodule("lm_head").weight.float()  # (vocab, d_model)
    # (W_U @ x)[t] - mean_v (W_U @ x)[v] == (W_U[t] - mean-row) @ x — fold the demean
    # into a single probe vector, then dot with each decoder direction.
    probe = (wu[token_id] - wu.mean(dim=0)) * gamma  # (d_model,)
    by_layer: dict[int, list[int]] = defaultdict(list)
    for L, i in feats:
        by_layer[L].append(i)
    out: dict[tuple[int, int], float] = {}
    for L, idxs in sorted(by_layer.items()):
        dec = tc.transcoders[L]._get_decoder_vectors(torch.tensor(idxs, device=device)).float()
        for i, w in zip(idxs, (dec @ probe).tolist(), strict=True):
            out[(L, i)] = float(w)
    return out


# ---------------------------------------------------------------------------
# Feature probing on arbitrary prompts (the polymer-reuse test)
# ---------------------------------------------------------------------------


@torch.no_grad()
def probe_features_on_prompt(model, tc, tokenizer, feats, prompt: str):
    """Raw activations of ``feats = [(layer, idx)]`` at every position of a raw prompt.

    Returns ``(acts, active_final, ids)`` where ``acts["L{L}f{i}"] = {"nonzero":
    [(pos, tok, act)], "final": act}`` and ``active_final = [(L, i, act)]`` for features
    active at the final position — the inject/suppress set for reuse interventions.

    The polymer-reuse doctrine (digit-split tokenization): Haiku predicts "995" as ONE
    token, so its graph sees both digits at once; Qwen3 emits digits one at a time, and
    the ``_6+_9 -> _5`` lookup can only fire at the citation's ONES moment — probe
    ``POLYMER_PROMPT + "99"`` (predicting the final '5'), not the bare prompt.
    """
    dev = next(model.parameters()).device
    ids = tokenize_raw(tokenizer, prompt, dev)
    toks = [tokenizer.decode([int(t)]) for t in ids[0]]
    layers = sorted({L for L, _ in feats})

    captured: dict[int, torch.Tensor] = {}

    def mk(layer: int):
        def hook(_m, inp, _o):
            captured[layer] = inp[0]

        return hook

    handles = [
        model.get_submodule(f"model.layers.{L}.mlp").register_forward_hook(mk(L)) for L in layers
    ]
    try:
        model(ids)
    finally:
        for h in handles:
            h.remove()

    acts: dict[str, dict] = {}
    active_final: list[tuple[int, int, float]] = []
    last = ids.shape[1] - 1
    for L in layers:
        enc = tc.transcoders[L].encode(captured[L])[0]  # (seq, d_t)
        for L2, f in feats:
            if L2 != L:
                continue
            row = enc[:, f]
            nz = [(i, toks[i], round(float(row[i]), 2)) for i in range(1, len(toks)) if row[i] > 0]
            acts[f"L{L}f{f}"] = {"nonzero": nz, "final": round(float(row[last]), 3)}
            if float(row[last]) > 0:
                active_final.append((L, f, float(row[last])))
    return acts, active_final, ids


# ---------------------------------------------------------------------------
# Introspection dialogue (paper: model narrates the carry algorithm it doesn't use)
# ---------------------------------------------------------------------------


@torch.no_grad()
def introspection_dialogue(
    model,
    tokenizer,
    a: int = 36,
    b: int = 59,
    max_new: int = 80,
    *,
    how_question: str = "Briefly, how did you get that?",
    do_sample: bool = False,
    temperature: float = 0.7,
):
    """Two-turn chat: answer a+b, then explain how.  Returns both responses.

    The answer turn is always greedy (it is the behavior under study); ``do_sample`` /
    ``temperature`` / ``how_question`` vary only the EXPLANATION turn — the paper's
    carry-narration claim should be checked over sampled phrasings, not one greedy shot.
    """
    device = next(model.parameters()).device
    msgs = [{"role": "user", "content": f"Answer in one word. What is {a}+{b}?"}]
    ids = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
    ).to(device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()

    msgs += [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": how_question},
    ]
    ids2 = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
    ).to(device)
    gen_kw: dict = {"do_sample": do_sample, "pad_token_id": tokenizer.eos_token_id}
    if do_sample:
        gen_kw["temperature"] = temperature
    out2 = model.generate(
        ids2, attention_mask=torch.ones_like(ids2), max_new_tokens=max_new, **gen_kw
    )
    explanation = tokenizer.decode(out2[0, ids2.shape[1] :], skip_special_tokens=True).strip()
    return {"answer": answer, "explanation": explanation, "how_question": how_question}


# ---------------------------------------------------------------------------
# Corpus scan: a feature's top-activating dataset examples (paper Fig A3)
# ---------------------------------------------------------------------------


def stream_c4_texts(n_docs: int = 2000, *, min_len: int = 200, max_len: int = 2000) -> list[str]:
    """First ``n_docs`` length-filtered documents of streaming C4-en (network needed)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts: list[str] = []
    for ex in ds:
        t = ex["text"]
        if min_len < len(t) < max_len:
            texts.append(t)
        if len(texts) >= n_docs:
            break
    return texts


@torch.no_grad()
def corpus_scan(
    model,
    tc,
    features,
    texts,
    tokenizer,
    *,
    max_tokens: int = 96,
    batch_size: int = 16,
    top_k: int = 12,
):
    """Top-activating text windows for each ``(layer, feature_idx)`` over ``texts``.

    Mirrors the paper's dataset-example panels (Fig A3): each hit records the snippet with
    the max-activation token marked.  Returns ``{(L, i): [(act, snippet), ...]}``.
    """
    device = next(model.parameters()).device
    layers = sorted({L for L, _ in features})
    hits: dict[tuple[int, int], list[tuple[float, str]]] = {(L, i): [] for L, i in features}

    captured: dict[int, torch.Tensor] = {}

    def mk(layer: int):
        def hook(_m, inp, _o):
            captured[layer] = inp[0]

        return hook

    handles = [
        model.get_submodule(f"model.layers.{L}.mlp").register_forward_hook(mk(L)) for L in layers
    ]
    try:
        for s in range(0, len(texts), batch_size):
            chunk = texts[s : s + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_tokens,
            ).to(device)
            model(enc.input_ids, attention_mask=enc.attention_mask)
            mask = enc.attention_mask.bool()
            for L in layers:
                acts = tc.transcoders[L].encode(captured[L])  # (B, seq, d_t)
                for L2, f in features:
                    if L2 != L:
                        continue
                    col = acts[..., f].masked_fill(~mask, 0.0)  # (B, seq)
                    vmax, pos = col.max(dim=1)
                    for k in range(len(chunk)):
                        v = float(vmax[k])
                        if v <= 0:
                            continue
                        p = int(pos[k])
                        toks = enc.input_ids[k]
                        lo, hi = max(0, p - 12), min(int(mask[k].sum()), p + 8)
                        before = tokenizer.decode(toks[lo:p])
                        at = tokenizer.decode(toks[p : p + 1])
                        after = tokenizer.decode(toks[p + 1 : hi])
                        hits[(L, f)].append((v, f"{before}⟦{at}⟧{after}"))
    finally:
        for h in handles:
            h.remove()
    for key in hits:
        hits[key] = sorted(hits[key], key=lambda t: -t[0])[:top_k]
    return hits
