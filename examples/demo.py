#!/usr/bin/env python3
"""End-to-end toolkit demo on one addition example (merges the former demos).

Walks the full machinery once, on a single `a+b=` prompt — every step the
interactive server UI (`llm-circuits serve`) exposes, plus the replacement-model
sanity checks:

  [1] load        — model + transcoders from the registry           (UI: Load)
  [2] replace     — original vs transcoder-replaced forward (KL/cos/top-1)
  [3] local       — local replacement model: error nodes + freezing => exact logits
  [4] graph       — build attribution graph -> prune                (UI: Build)
  [5] reprune     — re-prune the same graph at looser/tighter thresholds (UI: Re-prune)
  [6] features    — attach labels, rank the answer-driving features
  [7] steer       — negative-steer top feature / whole supernode    (UI: Steer)
  [8] sweep       — constrained-patching end-layer sweep            (UI: Sweep)
  [9] curve       — progressive (cumulative) steering curve
  [10] explorer   — self-contained interactive graph-explorer HTML  (UI: graph view)

The 0.6B model cannot add reliably, so the demo defaults to Qwen3-4b (bf16) and
first searches a few prompt formats for one the model actually solves — run it on
a GPU node (`sbatch hpc/run_demo.sbatch`) or pass ``--size``/``--problem`` knobs.
Artifacts (JSON graph, explorer HTML, curve PNGs) land in ``artifacts/demo/``.

Usage:
    uv run python examples/demo.py                 # full walkthrough, Qwen3-4b
    uv run python examples/demo.py --size 1.7b --problem 2+3
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
    run_progressive_intervention,
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

# Prompt formats, simplest first (we prefer the cleanest circuit the model solves).
FORMATS: list[tuple[str, str]] = [
    ("plain", "{a}+{b}="),
    ("spaces", "{a} + {b} = "),
    ("sum_is", "The sum of {a} and {b} is "),
    ("question", "What is {a}+{b}? Answer with just the number."),
    ("fewshot", "1+1=2\n3+2=5\n7+1=8\n{a}+{b}="),
]
FALLBACK_PROBLEMS = [(2, 3), (3, 4), (2, 4), (4, 5), (1, 2)]
N_TOP_FEATURES = 8  # supernode size for steering/sweep/curve


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
    args = ap.parse_args()

    device = default_device()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    out_dir = artifacts_dir() / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ [1] load
    banner(f"[1/10] load — Qwen3-{args.size} + transcoders ({args.dtype}, {device})")
    model, tokenizer = load_qwen3(args.size, dtype_str=args.dtype, device_map=device)
    model.eval()
    loaded = load_transcoder(f"qwen3-{args.size}", device=device, dtype=dtype)
    tc, repo_id = loaded.transcoder, loaded.repo_id
    print(f"transcoders: {repo_id} ({len(tc)} layers)")

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
    banner("[2/10] replace — original vs transcoder-replaced forward")
    cmp_res = compare_models(model, tc, input_ids, n_bos_tokens=n_bos)
    kl = cmp_res.kl_divergence[n_bos:]
    cos = cmp_res.cosine_similarity[n_bos:]
    print(
        f"per-position (non-BOS): KL mean={kl.mean():.4f} max={kl.max():.4f} | "
        f"logit cosine mean={cos.mean():.4f} | "
        f"top1 agreement={cmp_res.top1_agreement[n_bos:].float().mean():.2%}"
    )

    # ------------------------------------------------------------------ [3] local
    banner("[3/10] local — local replacement model (error nodes + frozen attn/LN)")
    lctx = run_local_replacement(model, tc, input_ids, n_bos_tokens=n_bos)
    diff = (lctx.logits.float() - lctx.original_logits.float()).abs().max().item()
    print(f"max |local - original| logit diff: {diff:.2e}  (error correction => ~exact)")

    # ------------------------------------------------------------------ [4] graph
    banner("[4/10] graph — build attribution graph -> prune (0.8 / 0.98)")
    t0 = time.time()
    graph = build_attribution_graph(model, tc, input_ids, n_bos_tokens=n_bos)
    pruned = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
    pg = pruned.graph
    print(
        f"{len(graph.nodes)} nodes / {len(graph.edges)} edges  ->  pruned "
        f"{len(pg.nodes)} nodes / {len(pg.edges)} edges  ({time.time() - t0:.0f}s)"
    )

    # ---------------------------------------------------------------- [5] reprune
    banner("[5/10] reprune — same graph, other thresholds (the UI's re-prune slider)")
    for nt, et in [(0.6, 0.9), (0.95, 0.99)]:
        p2 = prune_graph(graph, node_threshold=nt, edge_threshold=et)
        print(
            f"node_threshold={nt:.2f} edge_threshold={et:.2f}: "
            f"{len(p2.graph.nodes)} nodes / {len(p2.graph.edges)} edges"
        )

    # --------------------------------------------------------------- [6] features
    banner("[6/10] features — labels + answer-driving supernode")
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

    # rank features by influence (stored per-node by graph_to_dict; use scores list)
    feat_ranked = sorted(
        (
            (score, nd)
            for score, nd in zip(pruned.influence_scores, pg.nodes, strict=True)
            if nd.node_type == "feature"
        ),
        key=lambda t: -t[0],
    )
    top = [nd for _, nd in feat_ranked[:N_TOP_FEATURES]]
    for nd in top[:5]:
        tops = (nd.label or {}).get("top_logits") or []
        print(f"  L{nd.layer:>2} f{nd.feature_idx:<7} pos={nd.position} top_logits={tops[:3]}")
    steers = [steer(nd.layer, nd.feature_idx, m=-2.0, position=nd.position) for nd in top]

    # ------------------------------------------------------------------ [7] steer
    banner(f"[7/10] steer — negative steering (m=-2) top-1 and all {len(steers)}")
    single = run_feature_intervention(model, tc, input_ids, steers[:1], n_bos_tokens=n_bos)
    p0, p1 = ablation_prob_effect(single, [answer_id])[answer_id]
    joint = run_feature_intervention(model, tc, input_ids, steers, n_bos_tokens=n_bos)
    _, pall = ablation_prob_effect(joint, [answer_id])[answer_id]
    new_top = tokenizer.decode(int(joint.ablated_logits[-1].argmax()))
    print(f"p({answer!r}): {p0:.3f} -> {p1:.3f} (top-1) -> {pall:.3f} (all; new top {new_top!r})")

    # ------------------------------------------------------------------ [8] sweep
    banner("[8/10] sweep — constrained-patching end layer (the paper's range knob)")
    sw = sweep_patch_end_layer(model, tc, input_ids, steers, answer_id, n_bos_tokens=n_bos)
    print(
        f"end layers {sw.end_layers[0]}..{sw.end_layers[-1]}: best (most suppressive) "
        f"= {sw.best_end_layer}  (delta logit {min(sw.delta_logits):+.2f})"
    )
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(sw.end_layers, sw.delta_probs, "o-")
    ax.axvline(sw.best_end_layer, color="red", ls="--", alpha=0.6)
    ax.set_xlabel("patch end layer")
    ax.set_ylabel(f"Δ p({answer!r})")
    ax.set_title(f"{prompt!r}: end-layer sweep")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png", dpi=130)

    # ------------------------------------------------------------------ [9] curve
    banner("[9/10] curve — progressive (cumulative) negative steering")
    res = run_progressive_intervention(model, tc, input_ids, steers, answer_id, n_bos_tokens=n_bos)
    print("p(answer) at k=0..N:", " ".join(f"{p:.2f}" for p in res.probs))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(res.n_ablated, res.probs, "o-")
    ax.set_xlabel("features negative-steered (cumulative)")
    ax.set_ylabel(f"p({answer!r})")
    ax.set_title(f"{prompt!r}: progressive steering")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "progressive.png", dpi=130)

    # --------------------------------------------------------------- [10] explorer
    banner("[10/10] explorer — interactive graph-explorer HTML")
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
    html_path = out_dir / "explorer.html"
    render_graph_explorer_html(gd, str(html_path), title=f"{prompt} -> {answer!r}")
    print(f"wrote {html_path}")

    print(
        f"\nDemo complete — artifacts in {out_dir}/ (graph.json, explorer.html, "
        f"sweep.png, progressive.png).\nFor the LIVE version of steps 4/5/7/8, "
        f"run the interactive UI:  llm-circuits serve"
    )


if __name__ == "__main__":
    main()
