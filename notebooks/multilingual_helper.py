"""Multilingual-circuit helpers for reproducing the paper's multilingual dive on Qwen3-4b.

https://transformer-circuits.pub/2025/attribution-graphs/biology.html (multilingual section)

The paper shows the antonym task ("the opposite of <word>") runs through a circuit with
three parts: an **operand** (the concept, e.g. "small"), an **operation** (antonym vs synonym),
and a **language** (detect the input language / emit in the target language).  The operation
and operand are largely **language-agnostic** (shared middle-layer features across EN/FR/ZH);
the language parts are language-specific (early input + late output).  Editing each part
independently transfers across languages.

Kept out of the core package (task-specific prompt formatting, the parallel-sentence overlap
metric, the feature-graft interventions).  Heavy functions run the model -> execute the
notebook on a GPU node via ``hpc/run_multilingual_notebook.sbatch``.

Format note: Qwen3-4b is an instruct model -- a bare completion ("The opposite of 'small' is")
makes it echo the prompt, but the instruction form below yields the answer as the FIRST
generated token (EN small->"large", FR petit->"Grand", ZH 小->"大"; all single tokens).
"""
# ruff: noqa: RUF001  (Chinese prompts legitimately use full-width punctuation)

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
from llm_circuits.circuits.interventions import FeatureIntervention, run_feature_intervention
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.transcoders.feature_labels import load_feature_labels
from llm_circuits.transcoders.registry import get_spec

N_BOS = 1  # Qwen3 chat template prepends one BOS-like token

LANGS = ("en", "fr", "zh")
LANG_NAME = {"en": "English", "fr": "French", "zh": "Chinese"}

# Instruction-form prompts (the answer is the first generated token).
ANTONYM_PROMPT = {
    "en": 'What is the opposite of "{w}"? Reply with only the word, nothing else.',
    "fr": 'Quel est le contraire de "{w}" ? Réponds avec un seul mot, rien d\'autre.',
    "zh": '"{w}"的反义词是什么？只回答一个词，不要其他内容。',
}
SYNONYM_PROMPT = {
    "en": 'Give one synonym of "{w}". Reply with only the word, nothing else.',
    "fr": 'Donne un synonyme de "{w}". Réponds avec un seul mot, rien d\'autre.',
    "zh": '"{w}"的近义词是什么？只回答一个词，不要其他内容。',
}
# Operand surface forms per language (concept -> {lang: word}).
WORD = {
    "small": {"en": "small", "fr": "petit", "zh": "小"},
    "hot": {"en": "hot", "fr": "chaud", "zh": "热"},
}


def antonym_prompt(concept: str, lang: str) -> str:
    return ANTONYM_PROMPT[lang].format(w=WORD[concept][lang])


def synonym_prompt(concept: str, lang: str) -> str:
    return SYNONYM_PROMPT[lang].format(w=WORD[concept][lang])


def tokenize(tokenizer, prompt: str, device) -> torch.Tensor:
    """Chat-template tokenisation identical to the graph builder (so positions line up)."""
    messages, _, tkw = prepare_messages(prompt, "qwen3", enable_thinking=False)
    return tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **tkw
    ).to(device)


# ---------------------------------------------------------------------------
# Task verification + attribution graphs
# ---------------------------------------------------------------------------


@torch.no_grad()
def model_answer(model, tokenizer, prompt: str, n_gen: int = 4) -> tuple[int, str, str]:
    """Greedy answer; returns (first_token_id, first_token_str, full_continuation)."""
    device = next(model.parameters()).device
    ids = tokenize(tokenizer, prompt, device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=n_gen,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    cont = out[0, ids.shape[1] :]
    return int(cont[0]), tokenizer.decode([int(cont[0])]), tokenizer.decode(cont).strip()


def build_graph(
    model,
    tc,
    tokenizer,
    prompt: str,
    *,
    max_feature_nodes: int = 8000,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    out_html=None,
    title: str = "",
):
    """Build -> prune -> label an attribution graph for the next (answer) token.

    Returns ``(pruned_dict, answer_token_id, input_ids)``; writes the explorer HTML if asked.
    """
    device = next(model.parameters()).device
    input_ids = tokenize(tokenizer, prompt, device)
    with torch.no_grad():
        answer_id = int(model(input_ids).logits[0, -1].argmax())

    graph = build_attribution_graph(
        model, tc, input_ids, n_bos_tokens=N_BOS, max_feature_nodes=max_feature_nodes
    )
    pruned = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    pg = pruned.graph

    repo_id = get_spec("qwen3-4b").transcoder_repo
    by_layer: dict[int, list[int]] = defaultdict(list)
    for nd in pg.nodes:
        if nd.node_type == "feature":
            by_layer[nd.layer].append(nd.feature_idx)
    labels: dict[tuple[int, int], dict] = {}
    for layer, idxs in by_layer.items():
        for fidx, lab in load_feature_labels(repo_id, layer, idxs).items():
            labels[(layer, fidx)] = lab.to_dict()
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
        render_graph_explorer_html(pruned_dict, out_html, title=title or prompt[:40])
    return pruned_dict, answer_id, input_ids


def feature_label(node) -> str:
    lab = node.get("label") or {}
    tops = lab.get("top_logits") or []
    base = f"L{node['layer']} f{node['feature_idx']}"
    return f"{base} ({'/'.join(map(str, tops[:3]))})" if tops else base


def answer_features(pruned_dict, top_k: int = 12):
    """Top-``top_k`` feature nodes by influence."""
    feats = [n for n in pruned_dict["nodes"] if n["node_type"] == "feature"]
    feats.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    return feats[:top_k]


def graph_features_at_position(pruned_dict, position: int, top_n: int = 10):
    """Influence-ranked feature nodes at ``position`` -> ``[(layer, idx, activation)]``.

    The operand/concept supernode for the operand swap: graph **influence** selects the
    features causally feeding the answer (the paper's criterion), unlike top-activation at a
    position which surfaces late generic features.  ``activation`` is the donor inject value.
    """
    feats = [
        n for n in pruned_dict["nodes"] if n["node_type"] == "feature" and n["position"] == position
    ]
    feats.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    return [(n["layer"], n["feature_idx"], float(n.get("activation", 0.0))) for n in feats[:top_n]]


def operand_token_pos(tokenizer, input_ids, surface: str) -> int | None:
    """Index of the (last) token covering the operand word in ``input_ids``.

    Robust to multi-token operands (e.g. French ``petit`` -> ``pet``+``it``): locate the
    operand substring in the decoded prompt and return the last token overlapping its span.
    """
    toks = [tokenizer.decode([int(t)]) for t in input_ids[0]]
    spans, s = [], ""
    for i, t in enumerate(toks):
        spans.append((len(s), len(s) + len(t), i))
        s += t
    start = s.rfind(surface.strip())
    if start < 0:
        return None
    end = start + len(surface.strip())
    hit = None
    for a, b, i in spans:
        if i >= N_BOS and a < end and b > start:  # token overlaps the operand span
            hit = i
    return hit


# ---------------------------------------------------------------------------
# Per-layer feature activations (for the cross-language overlap metric)
# ---------------------------------------------------------------------------


def _capture_mlp_inputs(model, tc, input_ids) -> dict[int, torch.Tensor]:
    """Run the model once, capturing each layer's MLP input (== transcoder input)."""
    captured: dict[int, torch.Tensor] = {}

    def mk(layer: int):
        def hook(_m, inp, _o):
            captured[layer] = inp[0].detach()

        return hook

    handles = [
        model.get_submodule(f"model.layers.{L}.mlp").register_forward_hook(mk(L))
        for L in range(len(tc))
    ]
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()
    return captured


def topk_features_by_layer(model, tc, prompt, tokenizer, k: int = 64) -> dict[int, set[int]]:
    """Top-``k`` features per layer by peak activation over the prompt's non-BOS positions."""
    device = next(model.parameters()).device
    ids = tokenize(tokenizer, prompt, device)
    cap = _capture_mlp_inputs(model, tc, ids)
    out: dict[int, set[int]] = {}
    with torch.no_grad():
        for L in range(len(tc)):
            f = tc.transcoders[L].encode(cap[L])[0]  # (seq, d_t)
            peak = f[N_BOS:].amax(0)  # (d_t,)
            out[L] = set(peak.topk(k).indices.tolist())
    return out


def _iou(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


# 20 parallel sentences (EN/FR/ZH) for the dataset-level overlap-by-layer analysis.
CORPUS = [
    {
        "en": "The cat sleeps on the warm bed.",
        "fr": "Le chat dort sur le lit chaud.",
        "zh": "猫睡在温暖的床上。",
    },
    {
        "en": "She bought fresh bread at the market.",
        "fr": "Elle a acheté du pain frais au marché.",
        "zh": "她在市场买了新鲜的面包。",
    },
    {
        "en": "The children play in the green park.",
        "fr": "Les enfants jouent dans le parc vert.",
        "zh": "孩子们在绿色的公园里玩耍。",
    },
    {
        "en": "Water boils at one hundred degrees.",
        "fr": "L'eau bout à cent degrés.",
        "zh": "水在一百度沸腾。",
    },
    {
        "en": "He reads a book every night.",
        "fr": "Il lit un livre chaque soir.",
        "zh": "他每晚读一本书。",
    },
    {"en": "The train arrives at noon.", "fr": "Le train arrive à midi.", "zh": "火车中午到达。"},
    {
        "en": "My brother lives in a big city.",
        "fr": "Mon frère habite dans une grande ville.",
        "zh": "我哥哥住在一个大城市。",
    },
    {
        "en": "The sun rises in the east.",
        "fr": "Le soleil se lève à l'est.",
        "zh": "太阳从东方升起。",
    },
    {
        "en": "They walked along the river.",
        "fr": "Ils ont marché le long de la rivière.",
        "zh": "他们沿着河边散步。",
    },
    {
        "en": "The teacher explained the lesson.",
        "fr": "Le professeur a expliqué la leçon.",
        "zh": "老师讲解了这节课。",
    },
    {
        "en": "Snow fell during the night.",
        "fr": "La neige est tombée pendant la nuit.",
        "zh": "夜里下了雪。",
    },
    {
        "en": "The doctor helped the sick man.",
        "fr": "Le médecin a aidé l'homme malade.",
        "zh": "医生帮助了那个生病的人。",
    },
    {
        "en": "We ate dinner together.",
        "fr": "Nous avons dîné ensemble.",
        "zh": "我们一起吃了晚饭。",
    },
    {
        "en": "The bird flew over the mountain.",
        "fr": "L'oiseau a volé au-dessus de la montagne.",
        "zh": "鸟飞过了山。",
    },
    {
        "en": "She sings a beautiful song.",
        "fr": "Elle chante une belle chanson.",
        "zh": "她唱了一首美丽的歌。",
    },
    {
        "en": "The old house stood on the hill.",
        "fr": "La vieille maison se dressait sur la colline.",
        "zh": "老房子矗立在山上。",
    },
    {"en": "He drives a fast car.", "fr": "Il conduit une voiture rapide.", "zh": "他开一辆快车。"},
    {
        "en": "The flowers bloom in spring.",
        "fr": "Les fleurs fleurissent au printemps.",
        "zh": "花在春天盛开。",
    },
    {
        "en": "I forgot my keys at home.",
        "fr": "J'ai oublié mes clés à la maison.",
        "zh": "我把钥匙忘在家里了。",
    },
    {
        "en": "The story made everyone laugh.",
        "fr": "L'histoire a fait rire tout le monde.",
        "zh": "这个故事让大家笑了。",
    },
]


def overlap_curves(model, tc, tokenizer, corpus=None, k: int = 64):
    """Per-layer feature-overlap (IoU) across language pairs, averaged over a parallel corpus.

    For each sentence (in each language) take the top-``k`` features per layer, then for each
    layer average the pairwise IoU across the corpus.  High IoU in the middle => shared
    multilingual features; low at the ends => language-specific.  Returns
    ``{pair: np.ndarray[n_layers]}`` for pairs ``en-fr``, ``en-zh``, ``fr-zh`` and ``mean``.
    """
    corpus = corpus or CORPUS
    n_layers = len(tc)
    pairs = [("en", "fr"), ("en", "zh"), ("fr", "zh")]
    acc = {f"{a}-{b}": np.zeros(n_layers) for a, b in pairs}
    for sent in corpus:
        per_lang = {lg: topk_features_by_layer(model, tc, sent[lg], tokenizer, k=k) for lg in LANGS}
        for a, b in pairs:
            for L in range(n_layers):
                acc[f"{a}-{b}"][L] += _iou(per_lang[a][L], per_lang[b][L])
    for key in acc:
        acc[key] /= len(corpus)
    acc["mean"] = np.mean([acc[f"{a}-{b}"] for a, b in pairs], axis=0)
    return acc


def plot_overlap_curves(curves, *, ax=None, title="Cross-language feature overlap by layer"):
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 4))
    for key, arr in curves.items():
        ax.plot(
            range(len(arr)),
            arr,
            label=key,
            lw=2.5 if key == "mean" else 1.4,
            color="k" if key == "mean" else None,
            alpha=1.0 if key == "mean" else 0.8,
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("top-k feature IoU")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if own:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Feature-graft interventions (operand / operation / language swap)
# ---------------------------------------------------------------------------


def position_supernode(model, tc, prompt, tokenizer, position, top_n: int = 10, *, max_layer=None):
    """Top-``top_n`` features by activation at ``position`` (layers ``< max_layer`` if given).

    ``position`` is an int index or ``"final"``.  Returns ``([(layer, idx, act)], input_ids)``.

    ``max_layer`` restricts the CANDIDATE POOL, not just the result: early-layer
    activations are much smaller than late-layer ones, so filtering a global top-``top_n``
    after the fact (as callers previously did) returns an empty set.
    """
    device = next(model.parameters()).device
    ids = tokenize(tokenizer, prompt, device)
    pos = ids.shape[1] - 1 if position == "final" else position
    cap = _capture_mlp_inputs(model, tc, ids)
    n_scan = len(tc) if max_layer is None else max(1, min(int(max_layer), len(tc)))
    cands: list[tuple[int, int, float]] = []
    with torch.no_grad():
        for L in range(n_scan):
            vec = tc.transcoders[L].encode(cap[L])[0][pos]  # (d_t,)
            v, i = vec.topk(top_n)
            cands.extend(
                (L, int(idx), float(act)) for act, idx in zip(v.tolist(), i.tolist(), strict=True)
            )
    cands.sort(key=lambda x: x[2], reverse=True)
    return cands[:top_n], ids


def run_graft(model, tc, recipient_ids, source_node, donor_node, position, *, scale: float = 1.0):
    """Ablate ``source_node`` and inject ``donor_node`` (donor acts x ``scale``) at ``position``.

    ``*_node`` are ``[(layer, idx, act)]`` lists; ``position`` is an int or ``"final"``.  The
    paper drives interventions well above the donor's natural activation (~6x) so the injected
    concept dominates — ``scale`` exposes that knob.  Donor features that also appear in the
    source are dropped from the ablation set so the inject isn't cancelled.
    """
    pos = recipient_ids.shape[1] - 1 if position == "final" else position
    donor_keys = {(L, idx) for (L, idx, _) in donor_node}
    ivs = [
        FeatureIntervention(L, idx, position=pos, m=-1.0)
        for (L, idx, _) in source_node
        if (L, idx) not in donor_keys
    ]
    ivs += [
        FeatureIntervention(L, idx, position=pos, value=scale * act) for (L, idx, act) in donor_node
    ]
    return run_feature_intervention(model, tc, recipient_ids, ivs, n_bos_tokens=N_BOS)


def lang_specific_final_features(
    model, tc, tokenizer, concept: str = "small", top_k: int = 40, keep: int = 12
):
    """Language-DETECTION supernode per language: features in that language's antonym-prompt
    final position that are NOT in the other languages' final-position top-``top_k``.

    Returns ``{lang: [(layer, idx, act)]}`` (the language-specific output features).
    """
    finals: dict[str, list[tuple[int, int, float]]] = {}
    sets: dict[str, set[tuple[int, int]]] = {}
    for lg in LANGS:
        node, _ = position_supernode(
            model, tc, antonym_prompt(concept, lg), tokenizer, "final", top_n=top_k
        )
        finals[lg] = node
        sets[lg] = {(L, i) for (L, i, _) in node}
    out: dict[str, list[tuple[int, int, float]]] = {}
    for lg in LANGS:
        others = set().union(*(sets[o] for o in LANGS if o != lg))
        spec = [(L, i, a) for (L, i, a) in finals[lg] if (L, i) not in others]
        out[lg] = spec[:keep]
    return out


def top_token_probs(logit_row, tokenizer, k: int = 6):
    p = logit_row.float().softmax(-1)
    v, i = p.topk(k)
    return [
        (tokenizer.decode([int(t)]).strip(), round(float(pv), 3))
        for pv, t in zip(v, i, strict=True)
    ]


# ---------------------------------------------------------------------------
# English-as-default: direct effect of a feature's decoder on each language's answer token
# ---------------------------------------------------------------------------


def direct_logit_effect(model, tc, layer: int, feature_idx: int, token_ids: dict[str, int]):
    """Direct effect of a feature's decoder output on each language's answer-token logit.

    Computes ``(g * W_dec[feature]) . (W_U[:, t] - mean_t W_U)`` -- the demeaned unembedding
    projection through the final RMSNorm weight g (elementwise ``*``).  The shared
    LN-scale/activation factors cancel across the three tokens, so the EN/FR/ZH magnitudes are
    directly comparable.  Returns ``{lang: float}``.
    """
    device = next(model.parameters()).device
    dec = (
        tc.transcoders[layer]
        ._get_decoder_vectors(torch.tensor([feature_idx], device=device))[0]
        .float()
    )  # (d_model,)
    gamma = model.get_submodule("model.norm").weight.float()
    wu = model.get_submodule("lm_head").weight.float()  # (vocab, d_model)
    contrib = wu @ (gamma * dec)  # (vocab,)
    demeaned = contrib - contrib.mean()
    return {lg: float(demeaned[t]) for lg, t in token_ids.items()}


# ---------------------------------------------------------------------------
# PAPER-EXACT protocols (biology.html, Multilingual Circuits)
# ---------------------------------------------------------------------------
# The paper's swap protocol differs from ``run_graft`` in two ways:
#   1. the SOURCE supernode is steered to a NEGATIVE multiple of its clean
#      activation (operation/language: -5x; operand: -0.5x) -- not just ablated;
#   2. the intervention is SWEPT along a strength axis (0 -> donor_max) and the
#      paper reports the crossover strength (~4x for the operation swap).
# Multiplier convention: paper multiples are MULTIPLICATIVE on the clean
# activation (M_paper), and our FeatureIntervention.m is additive-delta, so
# m = M_paper - 1 (e.g. -5x  ->  m=-6).

# Paper endpoint strengths per swap kind: (source_mult, donor_mult).
PAPER_SWAP_STRENGTHS = {
    "operation": (-5.0, 6.0),
    "operand": (-0.5, 1.5),
    "language": (-5.0, 6.0),
}


def paper_swap(
    model,
    tc,
    recipient_ids,
    source_node,
    donor_node,
    position,
    *,
    source_mult: float,
    donor_mult: float,
):
    """One paper-protocol swap: source at ``source_mult x clean``, donor at ``donor_mult x donor``.

    ``source_node``/``donor_node`` are ``[(layer, idx, act)]`` lists (``act`` = clean/donor
    activation); ``position`` is an int or ``"final"``.  Donor features also present in the
    source are dropped from the suppression set so the inject isn't cancelled.
    """
    pos = recipient_ids.shape[1] - 1 if position == "final" else position
    donor_keys = {(L, idx) for (L, idx, _) in donor_node}
    ivs = [
        FeatureIntervention(L, idx, position=pos, m=source_mult - 1.0)
        for (L, idx, _) in source_node
        if (L, idx) not in donor_keys
    ]
    ivs += [
        FeatureIntervention(L, idx, position=pos, value=donor_mult * act)
        for (L, idx, act) in donor_node
    ]
    return run_feature_intervention(model, tc, recipient_ids, ivs, n_bos_tokens=N_BOS)


def paper_swap_sweep(
    model,
    tc,
    recipient_ids,
    source_node,
    donor_node,
    position,
    tokenizer,
    *,
    kind: str,
    baseline_token: int,
    expected_token: int,
    n_steps: int = 13,
):
    """Sweep the swap strength 0 -> donor_max (the paper's Fig B3/B4 line charts).

    At strength ``s`` the donor is injected at ``s x donor_act`` and the source is steered to
    ``(source_max/donor_max)*s x clean`` -- a proportional ramp that hits the paper's quoted
    endpoint pair (e.g. -5x/+6x) exactly at ``s = donor_max``.  Tracks the probability of the
    ``baseline_token`` (clean answer) and ``expected_token`` (the swapped task's answer) at
    every step and reports the **crossover** = smallest s where P(expected) > P(baseline)
    (paper: ~4x for the operation swap, consistent across languages).

    Returns a dict with ``strengths``, ``p_baseline``, ``p_expected``, ``top_tokens`` (per
    step), and ``crossover`` (None if never crossed).
    """
    src_max, don_max = PAPER_SWAP_STRENGTHS[kind]
    strengths = [don_max * i / (n_steps - 1) for i in range(n_steps)]
    p_base, p_exp, tops = [], [], []
    for s in strengths:
        if s == 0.0:
            res = run_feature_intervention(model, tc, recipient_ids, [], n_bos_tokens=N_BOS)
            row = res.baseline_logits[-1]
        else:
            res = paper_swap(
                model,
                tc,
                recipient_ids,
                source_node,
                donor_node,
                position,
                source_mult=1.0 + (src_max - 1.0) * (s / don_max),
                donor_mult=s,
            )
            row = res.ablated_logits[-1]
        probs = row.float().softmax(-1)
        p_base.append(float(probs[baseline_token]))
        p_exp.append(float(probs[expected_token]))
        tops.append(top_token_probs(row, tokenizer, k=4))
    crossover = next(
        (s for s, pb, pe in zip(strengths, p_base, p_exp, strict=True) if pe > pb), None
    )
    return {
        "kind": kind,
        "strengths": strengths,
        "p_baseline": p_base,
        "p_expected": p_exp,
        "top_tokens": tops,
        "crossover": crossover,
    }


def plot_swap_sweeps(results_by_lang, tokenizer, token_strs, *, title, ax_row=None):
    """Paper-style probability-vs-strength panels, one per language (Fig B3/B4/B5)."""
    langs = list(results_by_lang)
    own = ax_row is None
    if own:
        _, ax_row = plt.subplots(1, len(langs), figsize=(4.2 * len(langs), 3.4))
    ax_row = np.atleast_1d(ax_row)
    for ax, lg in zip(ax_row, langs, strict=True):
        r = results_by_lang[lg]
        ax.plot(r["strengths"], r["p_baseline"], "o-", label=f"baseline {token_strs[lg][0]!r}")
        ax.plot(r["strengths"], r["p_expected"], "s-", label=f"expected {token_strs[lg][1]!r}")
        if r["crossover"] is not None:
            ax.axvline(r["crossover"], color="red", ls="--", alpha=0.6)
            ax.text(r["crossover"], 0.5, f" x{r['crossover']:.1f}", color="red", fontsize=8)
        ax.set_title(LANG_NAME[lg])
        ax.set_xlabel("intervention strength (x donor act)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    ax_row[0].set_ylabel("next-token probability")
    if own:
        plt.suptitle(title)
        plt.tight_layout()
    return ax_row


def early_language_detection_supernode(
    model,
    tc,
    tokenizer,
    concept: str = "small",
    *,
    max_layer_frac: float = 0.34,
    top_k: int = 40,
    keep: int = 12,
):
    """Language-detection supernodes per the paper: EARLY-layer, final-token features unique
    to each language (the paper swaps 'open-quote-in-language-X' / 'beginning-of-document-
    in-language-Y' features, which live early in the model).

    Like :func:`lang_specific_final_features` but restricted to layers
    ``< max_layer_frac * n_layers``.  Returns ``{lang: [(layer, idx, act)]}``.
    """
    n_layers = len(tc)
    lmax = max(1, int(n_layers * max_layer_frac))
    finals: dict[str, list[tuple[int, int, float]]] = {}
    sets: dict[str, set[tuple[int, int]]] = {}
    for lg in LANGS:
        # max_layer restricts the candidate pool itself — a global top-k is dominated by
        # late-layer activations and filtering it to early layers yields an EMPTY set.
        node, _ = position_supernode(
            model,
            tc,
            antonym_prompt(concept, lg),
            tokenizer,
            "final",
            top_n=top_k,
            max_layer=lmax,
        )
        finals[lg] = node
        sets[lg] = {(L, i) for (L, i, _) in node}
    out: dict[str, list[tuple[int, int, float]]] = {}
    for lg in LANGS:
        others = set().union(*(sets[o] for o in LANGS if o != lg))
        out[lg] = [(L, i, a) for (L, i, a) in finals[lg] if (L, i) not in others][:keep]
    return out


# ---------------------------------------------------------------------------
# PAPER-EXACT overlap: IOU of features active anywhere in context + baseline
# ---------------------------------------------------------------------------


def active_features_by_layer(model, tc, prompt, tokenizer) -> dict[int, set[int]]:
    """Features ACTIVE ANYWHERE in the (non-BOS) context, per layer — the paper's set
    definition for the cross-lingual IOU (vs. our earlier top-k-per-layer proxy)."""
    device = next(model.parameters()).device
    ids = tokenize(tokenizer, prompt, device)
    cap = _capture_mlp_inputs(model, tc, ids)
    out: dict[int, set[int]] = {}
    with torch.no_grad():
        for L in range(len(tc)):
            f = tc.transcoders[L].encode(cap[L])[0]  # (seq, d_t); JumpReLU zeros inactive
            active = (f[N_BOS:] > 0).any(0).nonzero(as_tuple=True)[0]
            out[L] = set(active.tolist())
    return out


def overlap_curves_paper(model, tc, tokenizer, corpus=None):
    """The paper's cross-lingual overlap analysis (Fig B7 protocol).

    For each paragraph and language pair, IOU per layer of the feature sets active anywhere
    in the context; plus the paper's **baseline**: the same IOU computed on UNRELATED
    paragraph pairs (paragraph i in language A vs paragraph (i+1) mod N in language B).
    Returns ``{pair: curve, f"{pair}-baseline": curve, "mean", "mean-baseline"}``.
    """
    corpus = corpus or CORPUS
    n_layers = len(tc)
    pairs = [("en", "fr"), ("en", "zh"), ("fr", "zh")]
    sets_by_lang: dict[str, list[dict[int, set[int]]]] = {lg: [] for lg in LANGS}
    for sent in corpus:
        for lg in LANGS:
            sets_by_lang[lg].append(active_features_by_layer(model, tc, sent[lg], tokenizer))
    n = len(corpus)
    out: dict[str, np.ndarray] = {}
    for a, b in pairs:
        main = np.zeros(n_layers)
        base = np.zeros(n_layers)
        for i in range(n):
            j = (i + 1) % n  # unrelated pairing for the baseline
            for L in range(n_layers):
                main[L] += _iou(sets_by_lang[a][i][L], sets_by_lang[b][i][L])
                base[L] += _iou(sets_by_lang[a][i][L], sets_by_lang[b][j][L])
        out[f"{a}-{b}"] = main / n
        out[f"{a}-{b}-baseline"] = base / n
    out["mean"] = np.mean([out[f"{a}-{b}"] for a, b in pairs], axis=0)
    out["mean-baseline"] = np.mean([out[f"{a}-{b}-baseline"] for a, b in pairs], axis=0)
    return out


# Longer parallel paragraphs (closer to the paper's "diverse paragraphs" than single
# sentences); appended to CORPUS for the paper-protocol overlap analysis.
PARAGRAPHS = [
    {
        "en": "The library opened early that morning. Students filled the reading room, "
        "and the smell of old paper hung in the air. By noon, every seat was taken.",
        "fr": "La bibliothèque a ouvert tôt ce matin-là. Les étudiants remplissaient la salle "
        "de lecture, et l'odeur du vieux papier flottait dans l'air. À midi, toutes les "
        "places étaient prises.",
        "zh": "那天早上图书馆很早就开门了。学生们坐满了阅览室，空气中弥漫着旧纸张的气味。到了中午，所有的座位都被占满了。",
    },
    {
        "en": "The storm arrived without warning. Fishermen pulled their boats onto the shore "
        "while dark clouds rolled over the harbor. Within an hour, the rain had flooded the streets.",
        "fr": "La tempête est arrivée sans prévenir. Les pêcheurs ont tiré leurs bateaux sur le "
        "rivage tandis que des nuages sombres roulaient sur le port. En une heure, la pluie "
        "avait inondé les rues.",
        "zh": "暴风雨毫无预警地来临了。渔民们把船拖上岸，乌云在港口上空翻滚。不到一个小时，雨水就淹没了街道。",
    },
    {
        "en": "Grandmother kept a small garden behind the house. Every summer she grew tomatoes, "
        "beans, and bright yellow sunflowers. The neighbors often stopped to admire it.",
        "fr": "Grand-mère entretenait un petit jardin derrière la maison. Chaque été, elle "
        "cultivait des tomates, des haricots et des tournesols jaune vif. Les voisins "
        "s'arrêtaient souvent pour l'admirer.",
        "zh": "祖母在房子后面种了一个小花园。每年夏天她都种西红柿、豆角和明黄色的向日葵。邻居们经常驻足欣赏。",
    },
    {
        "en": "The old clockmaker repaired watches for fifty years. His hands stayed steady even "
        "as his eyes grew weak. People brought him timepieces from across the country.",
        "fr": "Le vieil horloger a réparé des montres pendant cinquante ans. Ses mains sont "
        "restées sûres même quand ses yeux ont faibli. Les gens lui apportaient des montres "
        "de tout le pays.",
        "zh": "老钟表匠修了五十年的表。即使视力渐渐衰退，他的双手依然稳健。人们从全国各地把钟表送到他这里。",
    },
    {
        "en": "The train crossed the mountains at dawn. Passengers pressed their faces to the "
        "windows as snow-covered peaks appeared in the pink morning light.",
        "fr": "Le train a traversé les montagnes à l'aube. Les passagers collaient leur visage "
        "aux fenêtres tandis que les sommets enneigés apparaissaient dans la lumière rose du matin.",
        "zh": "火车在黎明时分穿越群山。乘客们把脸贴在车窗上，看着白雪覆盖的山峰出现在粉色的晨光中。",
    },
    {
        "en": "The market square filled with vendors before sunrise. Farmers arranged fruit in "
        "neat pyramids while bakers unloaded warm bread. By eight, the square buzzed with buyers.",
        "fr": "La place du marché s'est remplie de vendeurs avant le lever du soleil. Les "
        "fermiers disposaient les fruits en pyramides soignées pendant que les boulangers "
        "déchargeaient du pain chaud. À huit heures, la place bourdonnait d'acheteurs.",
        "zh": "日出之前，集市广场上就挤满了摊贩。农民们把水果摆成整齐的金字塔，面包师们卸下热腾腾的面包。到八点钟，广场上已经挤满了买东西的人。",
    },
    {
        "en": "The young scientist checked her results three times. The numbers pointed to the "
        "same surprising conclusion each time. She sat back and stared at the screen in silence.",
        "fr": "La jeune scientifique a vérifié ses résultats trois fois. Les chiffres menaient "
        "chaque fois à la même conclusion surprenante. Elle s'est adossée et a fixé l'écran "
        "en silence.",
        "zh": "年轻的科学家把她的结果核对了三遍。每一次数字都指向同一个令人惊讶的结论。她靠在椅背上，默默地盯着屏幕。",
    },
    {
        "en": "The theater dimmed its lights as the orchestra tuned. A hush fell over the "
        "audience when the conductor raised his baton. The first notes filled the hall like a wave.",
        "fr": "Le théâtre a baissé ses lumières pendant que l'orchestre s'accordait. Un silence "
        "est tombé sur le public quand le chef d'orchestre a levé sa baguette. Les premières "
        "notes ont rempli la salle comme une vague.",
        "zh": "剧院的灯光渐渐暗下来，乐队开始调音。指挥举起指挥棒时，观众席一片寂静。第一个音符如波浪般充满了大厅。",
    },
]
