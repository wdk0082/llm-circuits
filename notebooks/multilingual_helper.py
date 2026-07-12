"""Multilingual-circuit helpers for reproducing the paper's multilingual dive on Qwen3-4b.

https://transformer-circuits.pub/2025/attribution-graphs/biology.html (multilingual section)

The paper shows the antonym task ("the opposite of <word>") runs through a circuit with
three parts: an **operand** (the concept, e.g. "small"), an **operation** (antonym vs synonym),
and a **language** (detect the input language / emit in the target language).  The operation
and operand are largely **language-agnostic** (shared middle-layer features across EN/FR/ZH);
the language parts are language-specific (early input + late output).  Editing each part
independently transfers across languages.

Kept out of the core package (task-specific prompt formatting, the parallel-sentence overlap
metric, the supernode-file-driven swap protocol).  Reproduction v2: all selection lives in
the reviewed ``supernodes/multilingual_<size>.json`` (explorer-export ingest, see
``build_supernodes.py``); :func:`load_supernodes` refuses anything unapproved, and the
``supernode_*`` helpers below turn the file's members (per-graph ``acts``/``positions``)
into constrained-patching interventions and %-readouts.  Heavy functions run the model ->
execute the notebook on a GPU node via ``hpc/run_multilingual_notebook.sbatch``.

Format note: Qwen3-4b is an instruct model -- a bare completion ("The opposite of 'small' is")
makes it echo the prompt, but the instruction form below yields the answer as the FIRST
generated token (EN small->"large", FR petit->"Grand", ZH 小->"大"; all single tokens).
"""
# ruff: noqa: RUF001  (Chinese prompts legitimately use full-width punctuation)

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager as _fm

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

# ZH tick/legend labels (大/小/冷) need a CJK font; matplotlib >=3.6 falls back per glyph
# across a **font.family list of concrete families** (the sans-serif alias list does NOT
# trigger fallback), so Latin text keeps the DejaVu look and CJK glyphs fill in.
# Resolution order: a system CJK font if one exists, else the Noto Sans CJK SC shipped
# by the ``mplfonts`` package (notebook dependency group) — GPU nodes rarely have CJK
# system fonts, which used to leave tofu boxes in the ZH sweep panels. Guarded: neither
# available -> unchanged rcParams (no findfont warnings).
_cjk_names = {f.name for f in _fm.fontManager.ttflist if "CJK" in f.name}
_cjk = (
    "Noto Sans CJK SC" if "Noto Sans CJK SC" in _cjk_names else next(iter(sorted(_cjk_names)), None)
)
if not _cjk:
    try:
        import pathlib as _pathlib

        import mplfonts as _mplfonts

        _otf = _pathlib.Path(_mplfonts.__file__).parent / "fonts" / "NotoSansCJKsc-Regular.otf"
        if _otf.exists():
            _fm.fontManager.addfont(str(_otf))
            _cjk = "Noto Sans CJK SC"
    except ImportError:
        pass
if _cjk:
    plt.rcParams["font.family"] = ["DejaVu Sans", _cjk]

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
# Operation-word surface forms per language: the token span the operation-swap donors
# inject at (multi-token in FR/ZH; resolve with build_supernodes.op_word_positions).
OP_WORD = {"en": "opposite", "fr": "contraire", "zh": "反义词"}


def antonym_prompt(concept: str, lang: str) -> str:
    return ANTONYM_PROMPT[lang].format(w=WORD[concept][lang])


def synonym_prompt(concept: str, lang: str) -> str:
    return SYNONYM_PROMPT[lang].format(w=WORD[concept][lang])


# The paper's EXACT raw completion prompts (biology digest §B.1): the final token is a
# content-bearing open quote — where the paper's open-quote-in-language-X detection
# features live. The chat-template prompts end on assistant-header tokens instead, so the
# raw forms are the format control for the language swap. Raw behavior on Qwen3-4B
# (probed 2026-07-09): antonym large/grand/大 @ 0.77/0.99/0.99 — task intact; synonym
# en=tiny, fr echoes pet(it) (echo NOT a chat artifact), zh 小/微 near-tie.
RAW_ANTONYM_PROMPT = {
    "en": 'The opposite of "{w}" is "',
    "fr": 'Le contraire de "{w}" est "',
    "zh": '"{w}"的反义词是"',
}
RAW_SYNONYM_PROMPT = {
    "en": 'A synonym of "{w}" is "',
    "fr": 'Un synonyme de "{w}" est "',
    "zh": '"{w}"的近义词是"',
}


def raw_antonym_prompt(concept: str, lang: str) -> str:
    return RAW_ANTONYM_PROMPT[lang].format(w=WORD[concept][lang])


def raw_synonym_prompt(concept: str, lang: str) -> str:
    return RAW_SYNONYM_PROMPT[lang].format(w=WORD[concept][lang])


def tokenize(tokenizer, prompt: str, device) -> torch.Tensor:
    """Chat-template tokenisation identical to the graph builder (so positions line up)."""
    messages, _, tkw = prepare_messages(prompt, "qwen3", enable_thinking=False)
    return tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **tkw
    ).to(device)


def tokenize_raw(tokenizer, prompt: str, device) -> torch.Tensor:
    """Raw-completion tokenisation with a prepended special token (the attention sink).

    Mirrors ``addition_helper.tokenize_raw``: transcoders can't reconstruct position 0,
    so raw prompts get a special token (eos/pad — Qwen3 has no BOS) prepended and every
    downstream consumer keeps ``n_bos_tokens=1``.
    """
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    special = tokenizer.bos_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id
    ids = torch.cat([torch.tensor([[special]], dtype=ids.dtype), ids], dim=1)
    return ids.to(device)


def _tokenize(tokenizer, prompt: str, device, raw: bool) -> torch.Tensor:
    return tokenize_raw(tokenizer, prompt, device) if raw else tokenize(tokenizer, prompt, device)


# ---------------------------------------------------------------------------
# Task verification + attribution graphs
# ---------------------------------------------------------------------------


@torch.no_grad()
def model_answer(
    model, tokenizer, prompt: str, n_gen: int = 4, *, raw: bool = False
) -> tuple[int, str, str]:
    """Greedy answer; returns (first_token_id, first_token_str, full_continuation)."""
    device = next(model.parameters()).device
    ids = _tokenize(tokenizer, prompt, device, raw)
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
    size_key: str = "4b",
    max_feature_nodes: int = 8000,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    out_html=None,
    title: str = "",
    raw: bool = False,
):
    """Build -> prune -> label an attribution graph for the next (answer) token.

    ``raw=True`` uses :func:`tokenize_raw` (paper-format completion prompts) instead of
    the chat template; ``n_bos_tokens=1`` holds either way (sink-token prepend).
    Returns ``(pruned_dict, answer_token_id, input_ids)``; writes the explorer HTML if asked.
    """
    device = next(model.parameters()).device
    input_ids = _tokenize(tokenizer, prompt, device, raw)
    with torch.no_grad():
        answer_id = int(model(input_ids).logits[0, -1].argmax())

    graph = build_attribution_graph(
        model, tc, input_ids, n_bos_tokens=N_BOS, max_feature_nodes=max_feature_nodes
    )
    pruned = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    pg = pruned.graph

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
        render_graph_explorer_html(pruned_dict, out_html, title=title or prompt[:40])
    return pruned_dict, answer_id, input_ids


def answer_features(pruned_dict, top_k: int = 12):
    """Top-``top_k`` feature nodes by influence."""
    feats = [n for n in pruned_dict["nodes"] if n["node_type"] == "feature"]
    feats.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    return feats[:top_k]


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


# ---------------------------------------------------------------------------
# Supernode %-readouts (Fig B3-B5 node annotations)
# ---------------------------------------------------------------------------


def supernode_readout_pct(
    res, node, position, *, ref: str = "baseline", patch_end_layer: int | None = None
):
    """Fig B3-B5-style supernode %-readout: **per-feature ratio first, then mean**.

    Each feature's ratio is ``steered_activation / reference * 100`` and the
    supernode number is the mean of those individual ratios (NOT the ratio of summed
    or averaged activations — a big feature must not drown out the others).

    ``node`` is ``[(layer, idx)]`` or ``[(layer, idx, act)]``; ``position`` is an int
    or ``"final"``.  ``ref`` picks each feature's denominator:

    * ``"baseline"`` — its clean activation on the recipient prompt (the paper's
      "% of baseline" for source/upstream/say-X supernodes);
    * ``"stored"`` — the ``act`` carried in the node list (its donor-prompt / own-
      prompt activation), for supernodes whose recipient baseline is ~0.

    **What the number means** (DEVLOG_EXTRA §2.3): readouts re-encode the perturbed
    MLP *inputs*, and a decoder delta at a feature's own layer only affects layers
    *after* it — so a steered feature's own readout is blind to its commanded steer
    (lowest-layer members of a steered supernode read ~clean / ~0).  For **steered**
    supernodes (sources, injected donors) the readout is therefore the network's
    *propagated response*, NOT a verification of the commanded value; only for
    supernodes **disjoint from the intervened sets** (downstream say-X, recruited
    counterparts) does it correspond to the paper's Fig B3-B5 node annotations
    (Fig B5's "new say-large-Y 76-105%" is a *downstream recruited* reading).

    Features whose reference is ~0 cannot form a ratio and are excluded (counted in
    ``n_skipped``).  Under **constrained patching** pass the run's ``patch_end_layer``:
    features at layers ``<= ell`` are clamped by the protocol, so their readout is not a
    network response — they are dropped from the mean and counted in ``n_pinned``
    (DEVLOG_EXTRA §3.1 step 4; the paper's annotations are downstream nodes, above ``ell``).
    Requires the swap to have run with ``readout_layers`` covering every layer in
    ``node``.  Returns ``{"mean_pct", "n_used", "n_skipped", "n_pinned", "per_feature"}``
    (``mean_pct`` is ``None`` when no feature has a usable reference).
    """
    per: dict[str, float | str | None] = {}
    ratios: list[float] = []
    skipped = 0
    pinned = 0
    for entry in node:
        L, i = entry[0], entry[1]
        act = float(entry[2]) if len(entry) > 2 else 0.0
        if patch_end_layer is not None and patch_end_layer >= L:
            per[f"L{L}f{i}"] = "pinned"
            pinned += 1
            continue
        if L not in res.ablated_features:
            raise KeyError(
                f"layer {L} missing from ablated_features — pass readout_layers "
                "covering every readout supernode to paper_swap/run_feature_intervention"
            )
        b, a = res.baseline_features[L], res.ablated_features[L]
        b2 = b[0] if b.dim() == 3 else b
        a2 = a[0] if a.dim() == 3 else a
        pos = b2.shape[0] - 1 if position == "final" else int(position)
        steered = float(a2[pos, i])
        reference = float(b2[pos, i]) if ref == "baseline" else act
        if abs(reference) < 1e-6:
            per[f"L{L}f{i}"] = None
            skipped += 1
            continue
        pct = steered / reference * 100.0
        per[f"L{L}f{i}"] = round(pct, 1)
        ratios.append(pct)
    mean_pct = round(sum(ratios) / len(ratios), 1) if ratios else None
    return {
        "mean_pct": mean_pct,
        "n_used": len(ratios),
        "n_skipped": skipped,
        "n_pinned": pinned,
        "per_feature": per,
    }


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
# PAPER-EXACT protocols (biology.html, Multilingual Circuits) — reproduction v2
# ---------------------------------------------------------------------------
# The paper's swap protocol: the SOURCE supernode is steered to a NEGATIVE multiple
# of its clean activation (operation/language: -5x; operand: -0.5x), the DONOR is
# injected at a positive multiple of its donor-prompt activation, and the strength
# is SWEPT 0 -> donor_max with the crossover reported (~4x for the operation swap).
# Multiplier convention: paper multiples are MULTIPLICATIVE on the clean
# activation (M_paper), and our FeatureIntervention.m is additive-delta, so
# m = M_paper - 1 (e.g. -5x  ->  m=-6).
#
# v2 sources/donors/readouts come from the REVIEWED supernode file: members carry
# per-graph ``acts``/``positions``, sources steer at each member's own node position
# in the recipient graph, donors inject value = mult x their stored donor-graph act.

# Paper endpoint strengths per swap kind: (source_mult, donor_mult).
PAPER_SWAP_STRENGTHS = {
    "operation": (-5.0, 6.0),
    "operand": (-0.5, 1.5),
    "language": (-5.0, 6.0),
}


def load_supernodes(path):
    """Load the REVIEWED multilingual supernode file (``build_supernodes.py
    --from-exports``) for the notebook.

    Refuses files that are not ``approved``, contain ``rejected`` members, or violate
    the multilingual disjointness invariant — the paper's supernodes are disjoint
    FEATURE sets, so a ``(layer, feature)`` may belong to at most one supernode
    anywhere across the pages (the addition loader's per-graph key is too weak here).
    Returns ``{name: supernode_dict}``; members carry per-graph ``acts``/``positions``.
    The notebook derives its intervention/readout sets ONLY from this file — selection
    lives in the reviewed artifact (explorer-export), not in notebook code.
    """
    doc = json.loads(Path(path).read_text())
    if not doc.get("approved"):
        raise ValueError(f"{path}: approved=false — review the supernode file first")
    owner: dict[tuple[int, int], str] = {}
    out: dict[str, dict] = {}
    for sn in doc["supernodes"]:
        for m in sn["members"]:
            if m.get("review") == "rejected":
                raise ValueError(
                    f"{path}: rejected member L{m['layer']}f{m['feature']} still in"
                    f" {sn['name']!r} members"
                )
            key = (m["layer"], m["feature"])
            if key in owner and owner[key] != sn["name"]:
                raise ValueError(
                    f"{path}: L{key[0]}f{key[1]} in both {owner[key]!r} and {sn['name']!r}"
                    " — supernodes are disjoint feature sets"
                )
            owner[key] = sn["name"]
        out[sn["name"]] = sn
    return out


def supernode_suppress_ivs(sn, graph, *, mult, fallback_pos):
    """Suppression list for a swap source: every member steered to ``mult x clean``
    (``m = mult - 1``) at its own node position in ``graph``; members with no recorded
    position there steer at ``fallback_pos`` instead.  The m convention makes the
    fallback a no-op where the feature is inactive (clean ~ 0 => delta ~ 0), but the
    member still counts toward ``l_max`` — the constrained sweep's floor.
    """
    return [
        FeatureIntervention(
            m["layer"],
            m["feature"],
            position=int((m.get("positions") or {}).get(graph, fallback_pos)),
            m=mult - 1.0,
        )
        for m in sn["members"]
    ]


def supernode_inject_ivs(sn, donor_graph, positions, *, mult):
    """Donor-injection list: ``value = mult x acts[donor_graph]`` at every position in
    ``positions`` (the recipient-side token span).  Members with no stored activation
    on ``donor_graph`` are skipped — there is no donor-prompt value to scale.  Returns
    ``(ivs, used)`` with ``used = [(layer, feature, stored_act)]`` for reporting.
    """
    ivs, used = [], []
    for m in sn["members"]:
        act = (m.get("acts") or {}).get(donor_graph)
        if act is None:
            continue
        used.append((m["layer"], m["feature"], float(act)))
        ivs.extend(
            FeatureIntervention(m["layer"], m["feature"], position=int(p), value=mult * float(act))
            for p in positions
        )
    return ivs, used


def swap_ivs_fn(
    source_sn,
    recipient_graph,
    source_fallback_pos,
    donor_sn,
    donor_graph,
    donor_positions,
    *,
    kind,
    strengths=None,
):
    """``ivs(s)`` for the paper ramp: at strength ``s`` (0 -> donor endpoint) the donor
    is injected at ``s x`` its stored donor-graph act and the source steered to
    ``1 + (src_max - 1) * (s / don_max) x`` clean — hitting the paper's endpoint pair
    exactly at ``s = don_max`` (operation -5x/+6x, operand -0.5x/+1.5x, language
    -5x/+6x).  Source and donor supernodes are disjoint by the file invariant, so no
    dedup is needed.

    ``strengths`` overrides the endpoint pair ``(src_max, don_max)`` — for
    beyond-paper extended ladders (multilingual_extra_operand_swap.ipynb).  An
    override that preserves the ratio ``(src_max - 1) / don_max`` keeps the SAME ramp
    line, so the ladder still passes through the paper endpoint on the way (operand:
    (-14, 15) extends (-0.5, 1.5) tenfold along the identical coupling).
    """
    src_max, don_max = strengths if strengths is not None else PAPER_SWAP_STRENGTHS[kind]

    def ivs(s: float):
        if s == 0.0:
            return []
        src_mult = 1.0 + (src_max - 1.0) * (s / don_max)
        sup = supernode_suppress_ivs(
            source_sn, recipient_graph, mult=src_mult, fallback_pos=source_fallback_pos
        )
        inj, _ = supernode_inject_ivs(donor_sn, donor_graph, donor_positions, mult=s)
        return sup + inj

    return ivs


def choose_swap_end_layer(model, tc, recipient_ids, endpoint_ivs, expected_token):
    """The paper's intervention-layer recipe for a swap: sweep the constrained-patching
    end layer ``ell`` over ``[l_max, n_layers-1]`` at the endpoint strengths and pick
    the ``ell`` that promotes the expected (swapped-in) answer most.

    Returns ``(ell, sweep)``; ``sweep.end_layers[0]`` is ``l_max`` (the last steered
    layer — when a supernode member reaches the final layers the sweep has little or no
    room and constrained patching degenerates toward the pure direct effect).
    """
    sweep = sweep_patch_end_layer(
        model, tc, recipient_ids, endpoint_ivs, expected_token, n_bos_tokens=N_BOS
    )
    return sweep.most_promoting_end_layer, sweep


def supernode_swap_sweep(
    model,
    tc,
    recipient_ids,
    ivs_fn,
    tokenizer,
    *,
    kind: str,
    baseline_token: int,
    expected_token: int,
    patch_end_layer: int,
    n_steps: int = 13,
    strengths=None,
):
    """Fig B3/B4/B5 strength sweep 0 -> the donor endpoint, every step under the
    paper's constrained patching at the fixed ``patch_end_layer`` (choose it first with
    :func:`choose_swap_end_layer`).  ``ivs_fn(s)`` builds the intervention list at
    strength ``s`` (:func:`swap_ivs_fn`); ``s = 0`` is the clean forward.  Tracks
    P(baseline answer) and P(expected swapped answer) and reports the **crossover** =
    smallest ``s`` with P(expected) > P(baseline) (paper: ~4x for the operation swap,
    consistent across languages).  ``strengths`` overrides the endpoint pair for
    beyond-paper extended ladders (must match the ``ivs_fn``'s own override).
    """
    _, don_max = strengths if strengths is not None else PAPER_SWAP_STRENGTHS[kind]
    strengths = [don_max * i / (n_steps - 1) for i in range(n_steps)]
    p_base, p_exp, tops = [], [], []
    for s in strengths:
        ivs = ivs_fn(s)
        res = run_feature_intervention(
            model,
            tc,
            recipient_ids,
            ivs,
            n_bos_tokens=N_BOS,
            patch_end_layer=patch_end_layer if ivs else None,
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
        "patch_end_layer": patch_end_layer,
    }


def supernode_readout(res, sn, graph, *, final_pos, ref_graph=None, patch_end_layer=None):
    """Fig B3-B5-style %-readout of a REVIEWED supernode, per-member node positions.

    Same semantics as :func:`supernode_readout_pct` — per-feature ratio first, then
    mean; ~0-reference features skipped; rows at layers ``<= patch_end_layer`` are
    pinned by the constrained protocol and dropped — but each member reads at ITS OWN
    node position in ``graph`` (``final_pos`` when it has none there), and ``ref_graph``
    selects the denominator: ``None`` = the member's clean baseline on the recipient
    (the paper's "% of baseline"); a graph name = its stored act there (falling back to
    its max stored act when that graph is absent), for supernodes with ~0 recipient
    baseline (recruited donors / say-X readouts).  Requires the run to have captured
    ``readout_layers`` covering every member layer above ``patch_end_layer``
    (:func:`readout_layers_above`).
    """
    per: dict[str, float | str | None] = {}
    ratios: list[float] = []
    skipped = 0
    pinned = 0
    for m in sn["members"]:
        L, i = m["layer"], m["feature"]
        pos = int((m.get("positions") or {}).get(graph, final_pos))
        key = f"L{L}f{i}@p{pos}"
        if patch_end_layer is not None and patch_end_layer >= L:
            per[key] = "pinned"
            pinned += 1
            continue
        if L not in res.ablated_features:
            raise KeyError(
                f"layer {L} missing from ablated_features — pass readout_layers covering"
                " every readout supernode layer above ell to run_feature_intervention"
            )
        b, a = res.baseline_features[L], res.ablated_features[L]
        b2 = b[0] if b.dim() == 3 else b
        a2 = a[0] if a.dim() == 3 else a
        steered = float(a2[pos, i])
        if ref_graph is None:
            reference = float(b2[pos, i])
        else:
            acts = m.get("acts") or {}
            reference = float(acts.get(ref_graph, max(acts.values(), default=0.0)))
        if abs(reference) < 1e-6:
            per[key] = None
            skipped += 1
            continue
        pct = steered / reference * 100.0
        per[key] = round(pct, 1)
        ratios.append(pct)
    mean_pct = round(sum(ratios) / len(ratios), 1) if ratios else None
    return {
        "mean_pct": mean_pct,
        "n_used": len(ratios),
        "n_skipped": skipped,
        "n_pinned": pinned,
        "per_feature": per,
    }


def readout_layers_above(sns, ell):
    """Sorted member layers of ``sns`` above the patch end layer — the layers
    ``run_feature_intervention`` must capture for the %-readouts (layers ``<= ell``
    are pinned by the protocol and never read)."""
    return sorted({m["layer"] for sn in sns for m in sn["members"] if m["layer"] > ell})


def print_readout_row(tag, row):
    """One-line print of ``{row_name: supernode_readout(...)}`` readout dicts."""

    def _fmt(v):
        parts = [f"n={v['n_used']}"]
        if v.get("n_skipped"):
            parts.append(f"{v['n_skipped']} skipped")
        if v.get("n_pinned"):
            parts.append(f"{v['n_pinned']} pinned")
        return f"{v['mean_pct']}% ({', '.join(parts)})"

    print(f"readout {tag}: " + ", ".join(f"{k.split('_pct')[0]} {_fmt(v)}" for k, v in row.items()))


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
        ax.set_title(LANG_NAME.get(lg, lg))
        ax.set_xlabel("intervention strength (x donor act)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    ax_row[0].set_ylabel("next-token probability")
    if own:
        plt.suptitle(title)
        plt.tight_layout()
    return ax_row


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
    Returns ``{pair: curve, f"{pair}-baseline": curve, "mean", "mean-baseline",
    "set-size"}`` — ``set-size`` is the mean active-set size per layer (over corpus x
    languages), the granularity check for cross-model IOU comparisons (IOU is
    mechanically sensitive to how many features fire, so same-recipe pairs should show
    comparable set sizes).
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
    out["set-size"] = np.array(
        [
            np.mean([len(sets_by_lang[lg][i][L]) for lg in LANGS for i in range(n)])
            for L in range(n_layers)
        ]
    )
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
