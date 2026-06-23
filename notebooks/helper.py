"""Addition-specific helpers for reproducing the paper's addition dive on Qwen3-4b.

https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-addition

These are deliberately kept *out* of the core ``llm_circuits`` package: they are
addition-task-specific (prompt formatting, the operand-grid feature sweep, the
mod-10/diagonal heatmap classifier) and don't generalise.

The heavy functions (``accuracy_grid``, ``feature_grids``) run the model, so the
notebook that uses them is executed on a GPU node via ``hpc/run_addition_notebook.sbatch``.
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
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.transcoders.feature_labels import load_feature_labels
from llm_circuits.transcoders.registry import get_spec

# Natural-language format Qwen3-4b handles best (bare "a+b=" makes the instruct model
# explain instead of answering); matches examples/addition_circuit.py.
PROMPT_TEMPLATE = "What is {a}+{b}? Answer with just the number."
N_BOS = 1  # Qwen3 chat template prepends one BOS-like token


def addition_prompt(a: int, b: int) -> str:
    return PROMPT_TEMPLATE.format(a=a, b=b)


def tokenize(tokenizer, prompt: str, device) -> torch.Tensor:
    """Chat-template tokenisation identical to the graph builder (so positions line up)."""
    messages, _, tkw = prepare_messages(prompt, "qwen3", enable_thinking=False)
    return tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **tkw
    ).to(device)


# ---------------------------------------------------------------------------
# Accuracy sweep
# ---------------------------------------------------------------------------


def accuracy_grid(model, tokenizer, a_vals, b_vals, batch_size: int = 256, n_gen: int = 4):
    """Greedy multi-token generation for every (a, b); returns (correct, predicted grids).

    Qwen3 tokenises numbers **digit-by-digit** ("95" -> ['9','5']), so we greedily generate
    ``n_gen`` tokens and compare the leading integer of the decoded continuation to
    ``str(a+b)``.  ``correct[i, j]`` is True iff they match.
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
            ids = tokenize(tokenizer, addition_prompt(a, b), device)
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


# ---------------------------------------------------------------------------
# Operand-grid feature activations (the headline visualisation)
# ---------------------------------------------------------------------------


def feature_grids(
    model, tc, features, a_vals, b_vals, tokenizer, batch_size: int = 256, *, probe: str = "first"
):
    """Activation of each feature over the full (a, b) grid.

    ``features`` is a list of ``(layer, feature_idx)``; ``probe`` picks **which position**
    the feature is measured at, mirroring the two circuits in :func:`build_addition_graph`:

    * ``"first"`` — plain prompt; **peak activation over the non-BOS positions** (captures
      the feature whether it fires on an operand or on the predict-first-digit position).
      Use for the **magnitude** circuit.
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
            ids = tokenize(tokenizer, addition_prompt(a, b), device)
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
                        if probe == "ones":
                            sel = feats[:, -1, :].float().cpu().numpy()  # predict-ones position
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
# Picking features to visualise + classifying their structure
# ---------------------------------------------------------------------------


def build_addition_graph(
    model,
    tc,
    tokenizer,
    a: int,
    b: int,
    *,
    target: str = "first",
    max_feature_nodes: int = 8000,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    out_html=None,
):
    """Build → prune → label an attribution graph for ``What is a+b?``.

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
    prompt_ids = tokenize(tokenizer, addition_prompt(a, b), device)
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
    pg = pruned.graph

    # Attach feature labels (top/bottom output logits per feature) for interpretability.
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
        prompt=addition_prompt(a, b),
        answer_token=tokenizer.decode(answer_id),
        tokens=[tokenizer.decode(t) for t in input_ids[0]],
        logit_token_strs=logit_strs,
    )
    if out_html is not None:
        render_graph_explorer_html(pruned_dict, out_html, title=f"{a}+{b}={a + b}")
    return pruned_dict, answer_id, input_ids


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


def periodicity_report(grid, a_vals, b_vals) -> dict:
    """Quantify the mod-10 / diagonal structure of an operand-grid heatmap.

    Returns metrics that distinguish the paper's feature families:

    * ``sum_ac10`` — autocorrelation of the sum-profile ``P(a+b)`` at lag 10.  High (>~0.4)
      ⇒ the feature fires on **periodic diagonals** (``a+b`` repeating every 10) — the mod-10
      *lookup* signature.  Near 0 ⇒ a single diagonal (magnitude).
    * ``s_conc`` / ``a_conc`` / ``b_conc`` — peak-to-mean ratio of activation grouped by
      ``(a+b)%10`` / ``a%10`` / ``b%10``.  >~1.5 ⇒ concentrated on one residue (ends-in-d).
    * ``s_std`` — spread of ``a+b`` over strongly-active cells (small ⇒ one diagonal).
    * ``label`` — derived family: ``mod10-sum(rN)`` (the lookup-table signature) /
      ``mod10-a(rN)`` / ``mod10-b(rN)`` / ``magnitude-diag`` / ``sparse`` (fires in <1% of
      cells) / ``mixed`` / ``inactive``.
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
        frac_on=frac_on,
    )
    if frac_on <= 0.01:
        rep["label"] = "sparse"
    elif sum_ac10 > 0.4 and s_conc > 1.3:
        rep["label"] = f"mod10-sum(r{s_top})"
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


def feature_label(node) -> str:
    """Short human label for a feature node (top output logits, if available)."""
    lab = node.get("label") or {}
    tops = lab.get("top_logits") or []
    base = f"L{node['layer']} f{node['feature_idx']}"
    return f"{base} ({'/'.join(map(str, tops[:3]))})" if tops else base
