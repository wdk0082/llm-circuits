"""PAPER-EXACT addition-dive machinery (biology.html §Addition + methods supplement).

Extends ``helper.py`` with the experiments the paper runs beyond the operand-grid
heatmaps: input-residue supernodes and their suppression (``_6`` -> expect a 9x-style
ones-digit shift, ``_9``, ``~30``/``~59`` magnitude inhibition), negative steering of
lookup vs sum features with the output-smear measurement, the journal-citation
(Polymer) reuse experiment, the intermediate-computation prompt, the introspection
dialogue, and a corpus scan for a feature's dataset examples.

Conventions: paper multiples are multiplicative (M_paper); ours additive-delta
(m = M_paper - 1). Paper -2x suppression == m=-3; sign-flip -1x == m=-2.
"""

from __future__ import annotations

import helper as H
import numpy as np
import torch

from llm_circuits.circuits.attribution_graph import build_attribution_graph
from llm_circuits.circuits.graph_explorer import render_graph_explorer_html
from llm_circuits.circuits.graph_pruning import graph_to_dict, prune_graph
from llm_circuits.circuits.interventions import FeatureIntervention, run_feature_intervention
from llm_circuits.transcoders.feature_labels import load_feature_labels
from llm_circuits.transcoders.registry import get_spec

# The paper's exact auxiliary prompts (Fig A4-A6), raw completions.
POLYMER_PROMPT = "K. Whang, etc. Polymer, 36, 837, 1"  # paper: completes "995"
INTERMEDIATE_PROMPT = "assert (4 + 5) * 3 =="  # paper: completes "27"

DIGIT_TOKENS = [str(d) for d in range(10)]


# ---------------------------------------------------------------------------
# Generic raw-prompt graph (Polymer / assert / calc share this)
# ---------------------------------------------------------------------------


def build_graph_raw(
    model,
    tc,
    tokenizer,
    prompt: str,
    *,
    size_key: str,
    max_feature_nodes: int = 8000,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    out_html=None,
    title: str = "",
):
    """Build -> prune -> label an attribution graph for a RAW (non-chat) prompt."""
    device = next(model.parameters()).device
    input_ids = H.tokenize_raw(tokenizer, prompt, device)
    with torch.no_grad():
        answer_id = int(model(input_ids).logits[0, -1].argmax())

    graph = build_attribution_graph(
        model, tc, input_ids, n_bos_tokens=H.N_BOS, max_feature_nodes=max_feature_nodes
    )
    pruned = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    pg = pruned.graph

    repo_id = get_spec(f"qwen3-{size_key}").transcoder_repo
    by_layer: dict[int, list[int]] = {}
    for nd in pg.nodes:
        if nd.node_type == "feature":
            by_layer.setdefault(nd.layer, []).append(nd.feature_idx)
    labels: dict[tuple[int, int], dict] = {}
    for layer, idxs in by_layer.items():
        for fidx, lab in load_feature_labels(repo_id, layer, idxs).items():
            labels[(layer, fidx)] = lab.to_dict()
    for nd in pg.nodes:
        if nd.node_type == "feature":
            nd.label = labels.get((nd.layer, nd.feature_idx))

    pruned_dict = graph_to_dict(
        pg,
        influence_scores=pruned.influence_scores,
        prompt=prompt,
        answer_token=tokenizer.decode(answer_id),
        tokens=[tokenizer.decode(t) for t in input_ids[0]],
        logit_token_strs={
            str(n.token_id): tokenizer.decode(n.token_id)
            for n in pg.nodes
            if n.node_type == "logit"
        },
    )
    if out_html is not None:
        render_graph_explorer_html(pruned_dict, out_html, title=title or prompt[:40])
    return pruned_dict, answer_id, input_ids


# ---------------------------------------------------------------------------
# Position-role supernodes (input _d / ~magnitude features live on operand tokens)
# ---------------------------------------------------------------------------


def digit_token_positions(tokenizer, input_ids, a: int, b: int) -> dict[str, list[int]]:
    """Positions of the operand digit tokens in a ``calc: a+b=`` prompt.

    Qwen3 tokenises digits one per token, so ``36+59`` -> tokens 3,6,+,5,9.  Returns
    ``{"a_digits": [...], "b_digits": [...], "eq": [pos_of_=]}`` (positions past N_BOS).
    """
    toks = [tokenizer.decode([int(t)]) for t in input_ids[0]]
    a_str, b_str = str(a), str(b)
    a_pos: list[int] = []
    b_pos: list[int] = []
    eq_pos: list[int] = []
    plus_seen = False
    for i, t in enumerate(toks):
        if i < H.N_BOS:
            continue
        s = t.strip()
        if s == "+":
            plus_seen = True
        elif s == "=":
            eq_pos.append(i)
        elif s.isdigit():
            (b_pos if plus_seen else a_pos).append(i)
    # keep only the trailing len(a_str)/len(b_str) digit tokens (guards against digits
    # appearing elsewhere in the prompt).
    return {"a_digits": a_pos[-len(a_str) :], "b_digits": b_pos[-len(b_str) :], "eq": eq_pos}


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
        rep = H.periodicity_report(grids[(L, i)], a_vals, b_vals)
        if predicate(rep):
            out.append((L, i, act))
    return out


# ---------------------------------------------------------------------------
# Interventions with feature readout + smear metrics
# ---------------------------------------------------------------------------


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
):
    """Steer ``feats`` (``[(layer, idx, act)]``) by ``m`` at ``position`` (None = the
    feature nodes' own recorded positions are NOT known here — pass an explicit position).

    Returns dict with top tokens (baseline/steered), the 0-9 digit distribution + smear
    metrics for both, and each ``readout`` feature's activation as % of baseline
    (``readout`` entries are ``(layer, feature_idx, position)``).
    """
    ivs = [FeatureIntervention(L, i, position=position, m=m) for (L, i, _) in feats]
    readout_layers = sorted({L for (L, _, _) in readout or []})
    res = run_feature_intervention(
        model, tc, input_ids, ivs, n_bos_tokens=H.N_BOS, readout_layers=readout_layers
    )
    row_b, row_s = res.baseline_logits[-1], res.ablated_logits[-1]
    report = {
        "m": m,
        "n_feats": len(feats),
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
            pct[f"L{L}f{i}@p{p}"] = {
                "baseline": b0,
                "steered": a0,
                "pct_of_baseline": (a0 / b0 * 100.0) if abs(b0) > 1e-6 else None,
            }
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
# Introspection dialogue (paper: model narrates the carry algorithm it doesn't use)
# ---------------------------------------------------------------------------


@torch.no_grad()
def introspection_dialogue(model, tokenizer, a: int = 36, b: int = 59, max_new: int = 80):
    """Two-turn chat: answer a+b, then explain how.  Returns both responses."""
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
        {"role": "user", "content": "Briefly, how did you get that?"},
    ]
    ids2 = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
    ).to(device)
    out2 = model.generate(
        ids2,
        attention_mask=torch.ones_like(ids2),
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    explanation = tokenizer.decode(out2[0, ids2.shape[1] :], skip_special_tokens=True).strip()
    return {"answer": answer, "explanation": explanation}


# ---------------------------------------------------------------------------
# Corpus scan: a feature's top-activating dataset examples (paper Fig A3)
# ---------------------------------------------------------------------------


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
