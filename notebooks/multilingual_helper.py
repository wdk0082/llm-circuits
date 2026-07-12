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


# The overlap corpus (v3.2, user decision 2026-07-12): 18 register-diverse parallel
# paragraphs (news, science, recipe, sports commentary, legal, weather, tech docs,
# finance, history, travel, product review, dialogue, fairy tale, business email,
# philosophy, health, music criticism, assembly manual), authored in EN and translated
# to FR/ZH in-repo — the paper's own recipe ("diverse paragraphs", Claude-generated
# translations). Replaces the earlier 28 short same-register items, whose stylistic
# uniformity inflated the unrelated-pair baseline (tenth-session probe).
CORPUS_DIVERSE = [
    {
        "en": "The city council approved the new transit budget on Tuesday after a heated "
        "three-hour debate. Opponents argued the plan favors downtown districts, while "
        "supporters pointed to decades of underinvestment in bus lines. Construction on "
        "the first two routes is expected to begin next spring.",
        "fr": "Le conseil municipal a approuvé mardi le nouveau budget des transports après "
        "un débat houleux de trois heures. Les opposants ont soutenu que le plan favorise "
        "les quartiers du centre-ville, tandis que les partisans ont rappelé des décennies "
        "de sous-investissement dans les lignes de bus. La construction des deux premières "
        "lignes devrait commencer au printemps prochain.",
        "zh": "市议会周二在经过三个小时的激烈辩论后批准了新的公共交通预算。反对者认为该计划偏向市中心各区，"
        "而支持者则指出公交线路数十年来投资不足。前两条线路的建设预计将于明年春天开工。",
    },
    {
        "en": "We measured the thermal conductivity of thin polymer films between 80 and "
        "300 kelvin. The results show a linear dependence on temperature below the glass "
        "transition, consistent with phonon-dominated transport. Above this threshold, "
        "conductivity saturates and becomes nearly independent of film thickness.",
        "fr": "Nous avons mesuré la conductivité thermique de films minces de polymère entre "
        "80 et 300 kelvins. Les résultats montrent une dépendance linéaire à la température "
        "en dessous de la transition vitreuse, ce qui est cohérent avec un transport dominé "
        "par les phonons. Au-delà de ce seuil, la conductivité sature et devient presque "
        "indépendante de l'épaisseur du film.",
        "zh": "我们测量了聚合物薄膜在80至300开尔文之间的热导率。结果表明，在玻璃化转变温度以下，"
        "热导率与温度呈线性关系，这与声子主导的输运机制一致。超过该阈值后，热导率趋于饱和，"
        "几乎不再随薄膜厚度变化。",
    },
    {
        "en": "Melt the butter in a heavy pan over medium heat, then add the chopped onions "
        "with a pinch of salt. Stir them every few minutes until they turn deep golden "
        "brown; this takes about forty minutes and cannot be rushed. Spread the "
        "caramelized onions over the dough and bake until the edges are crisp.",
        "fr": "Faites fondre le beurre dans une poêle épaisse à feu moyen, puis ajoutez les "
        "oignons émincés avec une pincée de sel. Remuez-les toutes les quelques minutes "
        "jusqu'à ce qu'ils prennent une couleur dorée foncée ; cela prend environ quarante "
        "minutes et ne peut pas être précipité. Étalez les oignons caramélisés sur la pâte "
        "et faites cuire jusqu'à ce que les bords soient croustillants.",
        "zh": "在厚底锅中用中火融化黄油，然后加入切碎的洋葱和一小撮盐。每隔几分钟翻炒一次，"
        "直到洋葱变成深金黄色；这大约需要四十分钟，急不得。将焦糖化的洋葱铺在面团上，烤至边缘酥脆。",
    },
    {
        "en": "With ninety seconds left, the visitors won a corner and sent their goalkeeper "
        "forward. The cross was cleared only as far as the edge of the box, where the "
        "captain met it with a first-time volley into the top corner. The home crowd fell "
        "silent as the away end erupted.",
        "fr": "À quatre-vingt-dix secondes de la fin, les visiteurs ont obtenu un corner et "
        "fait monter leur gardien. Le centre n'a été dégagé que jusqu'à l'entrée de la "
        "surface, où le capitaine l'a repris de volée dans la lucarne. Le public local "
        "s'est tu tandis que le parcage visiteur explosait.",
        "zh": "比赛还剩九十秒时，客队获得一个角球，并让门将压上参与进攻。传中球只被解围到禁区边缘，"
        "队长迎球凌空抽射，皮球直挂死角。主场观众鸦雀无声，客队球迷区则瞬间沸腾。",
    },
    {
        "en": "The tenant shall notify the landlord in writing of any defect within fourteen "
        "days of its discovery. Failure to provide timely notice releases the landlord "
        "from liability for consequential damages, except where the defect poses an "
        "immediate risk to health or safety. Repairs must then be completed within a "
        "reasonable period.",
        "fr": "Le locataire doit notifier par écrit au propriétaire tout défaut dans les "
        "quatorze jours suivant sa découverte. À défaut de notification dans les délais, "
        "le propriétaire est dégagé de toute responsabilité pour les dommages indirects, "
        "sauf si le défaut présente un risque immédiat pour la santé ou la sécurité. Les "
        "réparations doivent alors être effectuées dans un délai raisonnable.",
        "zh": "承租人应在发现任何缺陷后十四天内以书面形式通知出租人。未及时通知的，"
        "出租人对间接损失不承担责任，但缺陷对健康或安全构成直接威胁的除外。此后，维修必须在合理期限内完成。",
    },
    {
        "en": "A slow-moving cold front will cross the region overnight, bringing heavy rain "
        "and gusty winds to coastal areas. Snow levels will drop to about eight hundred "
        "meters by morning, with ten to twenty centimeters expected above that elevation. "
        "Travelers should expect delays on mountain passes through Thursday.",
        "fr": "Un front froid peu mobile traversera la région pendant la nuit, apportant de "
        "fortes pluies et des rafales de vent sur les zones côtières. La limite pluie-neige "
        "descendra vers huit cents mètres d'ici le matin, avec dix à vingt centimètres "
        "attendus au-dessus de cette altitude. Les voyageurs doivent s'attendre à des "
        "retards sur les cols de montagne jusqu'à jeudi.",
        "zh": "一股移动缓慢的冷锋将在夜间过境，给沿海地区带来强降雨和阵风。到早晨，雪线将降至约八百米，"
        "该海拔以上预计有十到二十厘米的降雪。周四之前，翻越山口的旅客应预留延误时间。",
    },
    {
        "en": "The cache client retries failed requests with exponential backoff, starting "
        "at fifty milliseconds and doubling up to a maximum of five seconds. Set the retry "
        "budget to zero to disable this behavior entirely. Note that idempotent operations "
        "are retried automatically, while writes require an explicit opt-in flag.",
        "fr": "Le client de cache réessaie les requêtes échouées avec un délai exponentiel, "
        "commençant à cinquante millisecondes et doublant jusqu'à un maximum de cinq "
        "secondes. Réglez le budget de nouvelles tentatives à zéro pour désactiver "
        "complètement ce comportement. Notez que les opérations idempotentes sont "
        "réessayées automatiquement, tandis que les écritures nécessitent un indicateur "
        "d'activation explicite.",
        "zh": "缓存客户端会以指数退避方式重试失败的请求，起始间隔为五十毫秒，逐次加倍，最长不超过五秒。"
        "将重试预算设为零可完全禁用此行为。请注意，幂等操作会自动重试，而写操作需要显式启用相应标志。",
    },
    {
        "en": "Shares of the shipping conglomerate fell six percent after the company cut "
        "its full-year guidance. Management blamed weaker container volumes on Asian "
        "routes and rising fuel costs. Analysts noted, however, that the dividend remains "
        "covered by free cash flow for now.",
        "fr": "L'action du conglomérat maritime a chuté de six pour cent après que "
        "l'entreprise a abaissé ses prévisions annuelles. La direction a mis en cause la "
        "baisse des volumes de conteneurs sur les routes asiatiques et la hausse des coûts "
        "du carburant. Les analystes ont toutefois noté que le dividende reste pour "
        "l'instant couvert par les flux de trésorerie disponibles.",
        "zh": "这家航运集团下调全年业绩指引后，股价下跌了百分之六。管理层将其归咎于亚洲航线集装箱运量疲软"
        "和燃油成本上升。不过分析师指出，目前股息仍有自由现金流的支撑。",
    },
    {
        "en": "The canal took nine years to dig and claimed hundreds of lives before the "
        "first barge passed through in 1832. Merchants who had once hauled grain over the "
        "mountains by mule could now move it to the coast in four days. Within a "
        "generation, the towns along the route had tripled in size.",
        "fr": "Le canal a demandé neuf ans de travaux et coûté des centaines de vies avant "
        "que la première péniche ne le franchisse en 1832. Les marchands qui "
        "transportaient autrefois le grain à dos de mulet par-dessus les montagnes "
        "pouvaient désormais l'acheminer jusqu'à la côte en quatre jours. En une "
        "génération, les villes situées le long du tracé avaient triplé de taille.",
        "zh": "这条运河挖了九年，夺去了数百人的生命，第一艘驳船才在1832年通过。曾经靠骡子翻山驮运粮食的商人，"
        "如今四天就能把货物运到海岸。不到一代人的时间，沿线城镇的规模扩大了两倍。",
    },
    {
        "en": "The old quarter is best explored on foot, ideally before the tour buses "
        "arrive at ten. Duck into the covered market for a bowl of noodle soup, then climb "
        "the bell tower for a view across the tiled roofs to the harbor. Most museums "
        "close on Mondays, so plan the citadel for the start of the week.",
        "fr": "Le vieux quartier s'explore de préférence à pied, idéalement avant l'arrivée "
        "des bus touristiques à dix heures. Faufilez-vous dans le marché couvert pour un "
        "bol de soupe de nouilles, puis montez au clocher pour admirer la vue sur les "
        "toits de tuiles jusqu'au port. La plupart des musées ferment le lundi, alors "
        "prévoyez la citadelle en début de semaine.",
        "zh": "老城区最适合步行游览，最好赶在旅游大巴十点到达之前。可以钻进有顶棚的市场喝一碗汤面，"
        "再登上钟楼，眺望层层瓦顶直至海港的景色。大多数博物馆周一闭馆，所以最好把城堡安排在一周的开头。",
    },
    {
        "en": "The kettle boils a full liter in just over two minutes, which is faster than "
        "anything else we tested. The lid hinge feels flimsy, though, and the handle gets "
        "uncomfortably warm during long pours. At this price we expected better materials, "
        "even if the performance is hard to fault.",
        "fr": "La bouilloire porte un litre entier à ébullition en un peu plus de deux "
        "minutes, ce qui est plus rapide que tout ce que nous avons testé. La charnière du "
        "couvercle semble toutefois fragile, et la poignée devient désagréablement chaude "
        "lors des longs versements. À ce prix, nous attendions de meilleurs matériaux, "
        "même si les performances sont difficiles à critiquer.",
        "zh": "这款电水壶烧开一整升水只需两分钟多一点，比我们测试过的任何产品都快。不过壶盖的铰链感觉不够结实，"
        "长时间倒水时手柄也会热得发烫。以这个价位，我们本期待更好的用料，尽管性能上确实无可挑剔。",
    },
    {
        "en": '"Did you hear the upstairs neighbors moved out?" she asked, setting down the '
        'groceries. "About time — I might finally sleep past six," he said, though he '
        "already missed the piano that used to drift through the ceiling on Sunday "
        "mornings.",
        "fr": "« Tu as su que les voisins du dessus ont déménagé ? » demanda-t-elle en "
        "posant les courses. « Ce n'est pas trop tôt : je vais peut-être enfin dormir "
        "passé six heures », dit-il, bien que le piano qui filtrait du plafond le "
        "dimanche matin lui manquât déjà.",
        "zh": "“你听说楼上的邻居搬走了吗？”她一边放下买来的菜一边问。"
        "“早该搬了——我总算能睡过六点了。”他嘴上这么说，心里却已经开始想念周日早晨从天花板上飘下来的钢琴声。",
    },
    {
        "en": "At the edge of the pine forest lived a clockmaker who had not spoken in "
        "seven years. Every night he wound the village clocks, and every morning the "
        "villagers found the hands pointing at hours that did not exist. One winter a "
        "child knocked at his door and asked him to repair a broken music box.",
        "fr": "À la lisière de la forêt de pins vivait un horloger qui n'avait pas parlé "
        "depuis sept ans. Chaque nuit, il remontait les horloges du village, et chaque "
        "matin les habitants trouvaient les aiguilles pointées vers des heures qui "
        "n'existaient pas. Un hiver, un enfant frappa à sa porte et lui demanda de "
        "réparer une boîte à musique cassée.",
        "zh": "松林边上住着一位钟表匠，他已经七年没有说过话了。每天夜里他为村里的钟上发条，"
        "每天清晨村民们都会发现指针指向并不存在的时刻。一年冬天，一个孩子敲响他的门，请他修理一只坏掉的音乐盒。",
    },
    {
        "en": "Following yesterday's call, I am attaching the revised timeline and the "
        "updated cost estimates. Please note that the vendor needs a signed purchase "
        "order by Friday to hold the current pricing. If any department has concerns, "
        "raise them in tomorrow's standup rather than by email.",
        "fr": "À la suite de notre appel d'hier, je joins le calendrier révisé et les "
        "estimations de coûts mises à jour. Veuillez noter que le fournisseur a besoin "
        "d'un bon de commande signé d'ici vendredi pour maintenir les prix actuels. Si un "
        "service a des objections, merci de les soulever lors du point de demain plutôt "
        "que par courriel.",
        "zh": "接昨天的电话会议，随信附上修订后的时间表和更新的成本估算。请注意，供应商需要在周五之前"
        "收到签署的采购订单，才能保住当前报价。如任何部门有异议，请在明天的站会上提出，而不要通过邮件。",
    },
    {
        "en": "To call a memory accurate is already to compare it with something that no "
        "longer exists. What we actually test is whether the memory coheres with "
        "documents, with other people's accounts, and with our expectations. Accuracy, in "
        "practice, is a verdict rendered by the present over the past.",
        "fr": "Dire qu'un souvenir est fidèle, c'est déjà le comparer à quelque chose qui "
        "n'existe plus. Ce que nous vérifions en réalité, c'est si le souvenir s'accorde "
        "avec des documents, avec les récits d'autrui et avec nos attentes. La fidélité, "
        "en pratique, est un verdict que le présent rend sur le passé.",
        "zh": "说一段记忆是准确的，就已经是在拿它与某种不复存在的东西作比较。我们实际检验的，"
        "是这段记忆能否与文件、他人的叙述以及我们的预期相吻合。所谓准确，实践中不过是现在对过去作出的裁决。",
    },
    {
        "en": "Adults should aim for at least one hundred fifty minutes of moderate "
        "activity per week, spread over several days. Short sessions count: three brisk "
        "ten-minute walks provide much of the benefit of a single long workout. People "
        "with heart conditions should consult a physician before starting any new "
        "program.",
        "fr": "Les adultes devraient viser au moins cent cinquante minutes d'activité "
        "modérée par semaine, réparties sur plusieurs jours. Les séances courtes "
        "comptent : trois marches rapides de dix minutes procurent une grande partie des "
        "bénéfices d'un long entraînement unique. Les personnes souffrant de troubles "
        "cardiaques devraient consulter un médecin avant de commencer tout nouveau "
        "programme.",
        "zh": "成年人每周应进行至少一百五十分钟的中等强度活动，并分散在数天进行。短时间的锻炼同样有效："
        "三次快步走十分钟，可以带来一次长时间锻炼的大部分益处。患有心脏疾病的人在开始任何新的锻炼计划前应咨询医生。",
    },
    {
        "en": "The quartet took the slow movement at a daringly quiet dynamic, letting the "
        "hall's silence become part of the texture. When the cello finally rose above the "
        "others, the effect was devastating precisely because nothing had prepared us for "
        "it. The finale, by contrast, felt rushed and almost apologetic.",
        "fr": "Le quatuor a joué le mouvement lent dans une nuance d'une discrétion "
        "audacieuse, laissant le silence de la salle faire partie de la texture. Quand le "
        "violoncelle s'est enfin élevé au-dessus des autres, l'effet fut bouleversant "
        "précisément parce que rien ne nous y avait préparés. Le finale, en revanche, a "
        "paru précipité et presque penaud.",
        "zh": "四重奏以大胆的弱音处理慢板乐章，让音乐厅的寂静融入音乐的肌理。当大提琴终于从其他声部之上升起时，"
        "效果之所以震撼，恰恰在于毫无预兆。相比之下，终曲则显得仓促，几乎带着歉意。",
    },
    {
        "en": "Attach the side panels to the base using the eight short screws, but do not "
        "tighten them fully yet. Slide the shelf into the middle groove, check that the "
        "unit stands level, and only then tighten all screws in a diagonal order. The "
        "back panel is nailed on last.",
        "fr": "Fixez les panneaux latéraux à la base à l'aide des huit vis courtes, mais "
        "sans les serrer complètement pour l'instant. Glissez l'étagère dans la rainure "
        "centrale, vérifiez que le meuble est bien de niveau, et alors seulement serrez "
        "toutes les vis en ordre diagonal. Le panneau arrière se cloue en dernier.",
        "zh": "用八颗短螺丝将侧板固定到底座上，但暂时不要完全拧紧。将搁板滑入中间的凹槽，确认柜体放置水平，"
        "然后再按对角顺序拧紧所有螺丝。背板最后用钉子固定。",
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
    patch_end_layer: int | None,
    n_steps: int = 13,
    strengths=None,
    freeze_attention: bool = True,
):
    """Fig B3/B4/B5 strength sweep 0 -> the donor endpoint, every step under the
    paper's constrained patching at the fixed ``patch_end_layer`` (choose it first with
    :func:`choose_swap_end_layer`).  ``ivs_fn(s)`` builds the intervention list at
    strength ``s`` (:func:`swap_ivs_fn`); ``s = 0`` is the clean forward.  Tracks
    P(baseline answer) and P(expected swapped answer) and reports the **crossover** =
    smallest ``s`` with P(expected) > P(baseline) (paper: ~4x for the operation swap,
    consistent across languages).  ``strengths`` overrides the endpoint pair for
    beyond-paper extended ladders (must match the ``ivs_fn``'s own override).

    ``patch_end_layer=None`` runs the fully-propagating variant, and there
    ``freeze_attention=False`` additionally lets attention patterns re-form — the
    probe for QK-mediated effects (a constrained range always freezes patterns, per
    circuit-tracer's coupling, so this knob only matters in propagate mode).
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
            freeze_attention=freeze_attention,
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
    corpus = corpus or CORPUS_DIVERSE
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
