#!/usr/bin/env python3
"""End-to-end toolkit demo on one addition example.

Walks the machinery the research actually leans on — the same calls
`notebooks/multilingual.ipynb` makes — on a single `a+b=` prompt:

  [1] load        — model + transcoders from the registry            (UI: Load)
  [2] replace     — the replacement model: raw transcoder swap is lossy, but with
                    error nodes + frozen attn/LN it reproduces the real logits
  [3] graph       — build attribution graph -> prune -> explorer HTML (UI: Build)
  [4] propagate   — intervention in *propagate mode* (patch_end_layer=None): the
                    perturbation flows through the real model, circuit-tracer semantics
  [5] constrained — intervention in *constrained mode*: activations <= L are pinned at
                    their perturbed values and the real model runs above L, with attention
                    patterns frozen. L is swept — the paper's layer-range knob   (UI: Sweep)

Steps 4 and 5 run the *same* steer on the *same* features, so the printout isolates
what constrained patching buys over letting the edit propagate. The features are chosen
*semantically* — the ones whose top output logits write the answer, below a depth cap —
which is the discipline the notebook's supernodes use, and the only selection under which
both modes bite (see the comment on SWEEP_MIN_LAYERS).

For the supernode/swap protocol the report is built on (semantic feature groups, donor
injection at absolute activations, coupled strength ladders), see
`notebooks/multilingual.ipynb`; that machinery lives in `notebooks/`, not the package.

The 0.6B model cannot add reliably, so the demo defaults to Qwen3-4b (bf16) and
first searches a few prompt formats for one the model actually solves — run it on
a GPU node (`sbatch hpc/run_demo.sbatch`) or pass ``--size``/``--problem`` knobs.
Artifacts (JSON graph, explorer HTML, sweep PNG) land in ``artifacts/demo/``.

Decoders are loaded **eagerly** (see the CLAUDE.md perf notes): the sweep alone runs one
intervention per end layer, and a lazy decoder re-reads ``W_dec`` from disk on every decode
(~9.3 s vs ~0.18 s per intervention). Eager 4b costs ~57 GB of transcoder weights on top of
the ~8 GB model, so on a smaller card pass ``--lazy-decoder`` to trade speed for VRAM.
The run is seeded (``--seed``, deterministic kernels) like the notebooks.

Usage:
    uv run python examples/demo.py                 # full walkthrough, Qwen3-4b
    uv run python examples/demo.py --size 1.7b --problem 2+3
    uv run python examples/demo.py --lazy-decoder  # low VRAM, much slower
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llm_circuits.circuits.attribution_graph import build_attribution_graph
from llm_circuits.circuits.graph_explorer import render_graph_explorer_html
from llm_circuits.circuits.graph_pruning import graph_to_dict, prune_graph
from llm_circuits.circuits.interventions import (
    ablation_prob_effect,
    run_feature_intervention,
    steer,
    sweep_patch_end_layer,
)
from llm_circuits.circuits.local_replacement_model import run_local_replacement
from llm_circuits.circuits.replacement_model import compare_models
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder
from llm_circuits.transcoders.feature_labels import load_feature_labels
from llm_circuits.utils import seed_everything

# Prompt formats, simplest first (we prefer the cleanest circuit the model solves).
FORMATS: list[tuple[str, str]] = [
    ("plain", "{a}+{b}="),
    ("spaces", "{a} + {b} = "),
    ("sum_is", "The sum of {a} and {b} is "),
    ("question", "What is {a}+{b}? Answer with just the number."),
    ("fewshot", "1+1=2\n3+2=5\n7+1=8\n{a}+{b}="),
]
FALLBACK_PROBLEMS = [(2, 3), (3, 4), (2, 4), (4, 5), (1, 2)]
N_TOP_FEATURES = 8  # supernode size for the steer
# Feature selection is *semantic*, not by influence rank — the same discipline the supernode
# pipeline in multilingual.ipynb uses, and for the same reason. Influence ranks the features
# that literally write the answer, and those sit in the last layers (L33-35 for 2+3=5);
# constrained patching sweeps range(max_steered_layer, n_layers), so a single L35 member
# leaves exactly one end layer and the sweep in [5] says nothing. Taking the answer-writing
# features *below* a depth cap keeps the causal punch and leaves a real range to sweep.
# (Measured alternatives: the most-influential features below the cap are junk token-level
# L0 features; the operand-position features are recovered above the patch and move nothing.)
SWEEP_MIN_LAYERS = 8  # end layers to leave above the deepest steered feature
NUM_WORDS = {
    3: "three", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}  # fmt: skip


def banner(step: str) -> None:
    print(f"\n{'=' * 74}\n{step}\n{'=' * 74}", flush=True)


def chat_ids(tokenizer, device, text: str):
    messages, n_bos, tkw = prepare_messages(text, "qwen3", enable_thinking=False)
    ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **tkw
    ).to(device)
    return ids, n_bos


@torch.no_grad()
def next_token(model, ids) -> int:
    return int(model(ids).logits[0, -1].argmax())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="4b", help="Qwen3 size key (0.6b cannot add)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--problem", default=None, help="e.g. '2+3' (default: first solved)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (pins deterministic kernels)")
    ap.add_argument(
        "--lazy-decoder",
        action="store_true",
        help="re-read W_dec from disk per decode: ~50x slower, but fits a small card",
    )
    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = default_device()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    out_dir = artifacts_dir() / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ [1] load
    banner(f"[1/5] load — Qwen3-{args.size} + transcoders ({args.dtype}, {device})")
    model, tokenizer = load_qwen3(args.size, dtype_str=args.dtype, device_map=device)
    model.eval()
    loaded = load_transcoder(
        f"qwen3-{args.size}", device=device, dtype=dtype, lazy_decoder=args.lazy_decoder
    )
    tc, repo_id = loaded.transcoder, loaded.repo_id
    decoder = "lazy" if args.lazy_decoder else "eager"
    print(f"transcoders: {repo_id} ({len(tc)} layers, {decoder} decoder, seed {args.seed})")

    # Pick a (format, problem) the model actually solves.
    problems = (
        [tuple(int(x) for x in args.problem.split("+"))] if args.problem else FALLBACK_PROBLEMS
    )
    chosen = None
    for fname, tmpl in FORMATS:
        for a, b in problems:
            ids, n_bos = chat_ids(tokenizer, device, tmpl.format(a=a, b=b))
            if tokenizer.decode(next_token(model, ids)).strip() == str(a + b):
                chosen = (fname, tmpl, a, b)
                break
        if chosen:
            break
    if not chosen:
        raise SystemExit("model solved none of the candidate problems — try a bigger --size")
    fname, tmpl, a, b = chosen
    prompt = tmpl.format(a=a, b=b)
    input_ids, n_bos = chat_ids(tokenizer, device, prompt)
    answer_id = next_token(model, input_ids)
    answer = tokenizer.decode(answer_id)
    print(f"studying {a}+{b}={a + b} (format {fname!r}) — model answers {answer!r} ✓")

    # ------------------------------------------------------------- [2] replacement
    banner("[2/5] replace — the replacement model (raw swap, then + error nodes)")
    cmp_res = compare_models(model, tc, input_ids, n_bos_tokens=n_bos)
    kl = cmp_res.kl_divergence[n_bos:]
    cos = cmp_res.cosine_similarity[n_bos:]
    print(
        f"raw transcoder swap (no error nodes): KL mean={kl.mean():.3f} | "
        f"logit cosine mean={cos.mean():.3f} | "
        f"top1 agreement={cmp_res.top1_agreement[n_bos:].float().mean():.0%}  <- lossy, as expected"
    )
    lctx = run_local_replacement(model, tc, input_ids, n_bos_tokens=n_bos)
    diff = (lctx.logits.float() - lctx.original_logits.float()).abs().max().item()
    scale = lctx.original_logits.float().abs().max().item()
    print(
        f"local replacement (+ error nodes, frozen attn/LN): max |Δ logit| = {diff:.2e} "
        f"on logits of scale {scale:.1f} (rel {diff / scale:.1e}, i.e. bf16 round-off)"
        f"  <- exact; this is what the graph is built on"
    )

    # ------------------------------------------------------------------ [3] graph
    banner("[3/5] graph — build attribution graph -> prune (0.8 / 0.98) -> explorer")
    t0 = time.time()
    graph = build_attribution_graph(model, tc, input_ids, n_bos_tokens=n_bos)
    pruned = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
    pg = pruned.graph
    print(
        f"{len(graph.nodes)} nodes / {len(graph.edges)} edges  ->  pruned "
        f"{len(pg.nodes)} nodes / {len(pg.edges)} edges  ({time.time() - t0:.0f}s)"
    )

    # Label the pruned features, then pick the supernode to steer.
    by_layer: dict[int, set[int]] = {}  # a set: a feature recurs at many positions
    for nd in pg.nodes:
        if nd.node_type == "feature":
            by_layer.setdefault(nd.layer, set()).add(nd.feature_idx)
    labels: dict[tuple[int, int], dict] = {}
    for layer, idxs in by_layer.items():
        for fidx, lab in load_feature_labels(repo_id, layer, sorted(idxs)).items():
            labels[(layer, fidx)] = lab.to_dict()
    for nd in pg.nodes:
        if nd.node_type == "feature":
            nd.label = labels.get((nd.layer, nd.feature_idx))

    gd = graph_to_dict(
        pg,
        influence_scores=pruned.influence_scores,
        prompt=prompt,
        model=f"qwen3-{args.size}",
        tokens=[tokenizer.decode(t) for t in input_ids[0]],
        n_bos_tokens=n_bos,
        answer_token=answer,
        logit_token_strs={
            str(n.token_id): tokenizer.decode(n.token_id)
            for n in pg.nodes
            if n.node_type == "logit"
        },
    )
    (out_dir / "graph.json").write_text(json.dumps(gd, indent=1))
    render_graph_explorer_html(gd, str(out_dir / "explorer.html"), title=f"{prompt} -> {answer!r}")
    print(f"wrote {out_dir / 'graph.json'} and explorer.html")

    feat_ranked = sorted(
        (
            (score, nd)
            for score, nd in zip(pruned.influence_scores, pg.nodes, strict=True)
            if nd.node_type == "feature"
        ),
        key=lambda t: -t[0],
    )
    layer_cap = len(tc) - SWEEP_MIN_LAYERS
    digits, word = str(a + b), NUM_WORDS.get(a + b, "")

    def writes_answer(nd) -> bool:
        tops = (nd.label or {}).get("top_logits") or []
        return any(digits in t or (word and word in t.lower()) for t in tops)

    top = [nd for _, nd in feat_ranked if nd.layer <= layer_cap and writes_answer(nd)]
    top = top[:N_TOP_FEATURES]
    if len(top) < 2:  # e.g. an answer with no labelled features — fall back to influence
        print(f"  only {len(top)} answer-writing features below L{layer_cap}; using influence rank")
        top = [nd for _, nd in feat_ranked if nd.layer <= layer_cap][:N_TOP_FEATURES]
    if not top:
        raise SystemExit(f"no pruned features at or below L{layer_cap} — loosen the prune")
    for nd in top[:5]:
        tops = (nd.label or {}).get("top_logits") or []
        print(f"  L{nd.layer:>2} f{nd.feature_idx:<7} pos={nd.position} top_logits={tops[:3]}")
    steers = [steer(nd.layer, nd.feature_idx, m=-2.0, position=nd.position) for nd in top]
    l_max = max(nd.layer for nd in top)
    print(
        f"steering {len(top)} features that write {answer.strip()!r}, "
        f"L{min(n.layer for n in top)}-L{l_max} (cap L{layer_cap} of {len(tc)}, "
        f"so [5] sweeps {len(tc) - l_max} end layers)"
    )

    # -------------------------------------------------------------- [4] propagate
    banner("[4/5] propagate — patch_end_layer=None: the edit flows through the real model")
    prop = run_feature_intervention(model, tc, input_ids, steers, n_bos_tokens=n_bos)
    p0, p_prop = ablation_prob_effect(prop, [answer_id])[answer_id]
    top_prop = tokenizer.decode(int(prop.ablated_logits[-1].argmax()))
    print(f"p({answer!r}): {p0:.3f} -> {p_prop:.3f}   (new top-1 {top_prop!r})")

    # ------------------------------------------------------------- [5] constrained
    banner(f"[5/5] constrained — pin activations <= L, real model above L; sweep L={l_max}..")
    sw = sweep_patch_end_layer(model, tc, input_ids, steers, answer_id, n_bos_tokens=n_bos)
    i_best = sw.end_layers.index(sw.best_end_layer)
    print(
        f"end layers {sw.end_layers[0]}..{sw.end_layers[-1]}: most suppressive L={sw.best_end_layer}"
        f"  p({answer!r}) {sw.baseline_prob:.3f} -> {sw.probs[i_best]:.3f}"
        f"  (Δ logit {sw.logits[i_best] - sw.baseline_logit:+.2f})"
    )
    print(
        f"propagate {p_prop:.3f}  vs  constrained@L{sw.best_end_layer} {sw.probs[i_best]:.3f}"
        f"   <- what the layer-range knob buys"
    )

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(sw.end_layers, sw.probs, "o-", label="constrained")
    ax.axhline(p_prop, color="grey", ls=":", label=f"propagate ({p_prop:.2f})")
    ax.axvline(sw.best_end_layer, color="red", ls="--", alpha=0.6)
    ax.set_xlabel("patch end layer L")
    ax.set_ylabel(f"p({answer!r})")
    ax.set_title(f"{prompt!r}: constrained-patching end-layer sweep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png", dpi=130)

    print(
        f"\nDemo complete — artifacts in {out_dir}/ (graph.json, explorer.html, sweep.png).\n"
        f"For the LIVE version of steps 3-5, run the interactive UI:  llm-circuits serve\n"
        f"For the supernode/swap protocol, see notebooks/multilingual.ipynb."
    )


if __name__ == "__main__":
    main()
