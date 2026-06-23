#!/usr/bin/env python3
"""Progressive negative-steering curves for the addition circuits.

Reuses the per-example attribution graphs saved by ``addition_circuit.py``: for
each, rank the answer-driving features by graph influence, then apply the paper's
**negative steering** (factor -1) to the top-1, top-2, … of them cumulatively via
**constrained patching** (freeze attention + LayerNorm + error nodes; inject the
change as a residual delta at the last steered layer), recording the answer
token's probability and logit at each step.

Negative steering is the paper's primary perturbation — a stronger, directional
test than ablation (which the paper argues is too weak and confounded by
reconstruction error). No graph rebuild needed (only intervention forwards).

Usage:
    uv run python examples/progressive_steering.py
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import torch

from llm_circuits.circuits.interventions import (
    FeatureIntervention,
    run_progressive_intervention,
    steer,
)
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_SIZE = "4b"
DTYPE_STR = "bf16"
MAX_FEATURES = 50  # cap; effectively ablates *all* feature nodes in each pruned graph
# ─────────────────────────────────────────────────────────────────────────────


def _answer_logit_index(d: dict) -> int | None:
    nodes = d["nodes"]
    logit_idxs = [i for i, n in enumerate(nodes) if n["node_type"] == "logit"]
    if not logit_idxs:
        return None
    lts = d.get("logit_token_strs", {})
    answer = d.get("answer_token")
    for i in logit_idxs:
        if lts.get(str(nodes[i]["token_id"])) == answer:
            return i
    return max(logit_idxs, key=lambda i: nodes[i]["activation"])


def _ordered_answer_features(d: dict) -> tuple[list[FeatureIntervention], int | None]:
    """Negative-steer interventions for answer features, ranked by graph influence.

    Influence is the power-iteration score propagated backward from the logits
    (concentrated on the predicted answer), so it ranks features by their effect
    on the answer through *all* paths — including indirect feature→feature→logit
    ones — not just direct feature→logit edges. Steering these (factor -1) breaks
    the answer with fewer features, giving cleaner decay curves.
    """
    ans_idx = _answer_logit_index(d)
    if ans_idx is None:
        return [], None
    nodes = d["nodes"]
    answer_token_id = nodes[ans_idx]["token_id"]
    feats = [n for n in nodes if n["node_type"] == "feature"]
    feats.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    steers = [
        steer(n["layer"], n["feature_idx"], m=-2.0, position=n["position"])
        for n in feats[:MAX_FEATURES]
    ]
    return steers, answer_token_id


def main() -> None:
    device = default_device()
    dtype = torch.float32 if DTYPE_STR == "fp32" else torch.bfloat16
    out_dir = artifacts_dir() / "addition_circuit"

    graph_files = sorted(out_dir.glob(f"addition_graph_qwen3-{MODEL_SIZE}_*plus*.json"))
    if not graph_files:
        print(f"No per-example graphs in {out_dir}. Run examples/addition_circuit.py first.")
        return
    print(f"Found {len(graph_files)} example graphs.")

    print(f"Loading Qwen3-{MODEL_SIZE} ({DTYPE_STR}) ...")
    model, tokenizer = load_qwen3(MODEL_SIZE, dtype_str=DTYPE_STR, device_map=device)
    model.eval()

    # Try eager decoders (no lazy per-forward disk reload = large speedup for the
    # many forwards below); fall back to lazy loading if it would OOM.
    lazy_decoder = True
    try:
        tc = load_transcoder(
            f"qwen3-{MODEL_SIZE}", device=device, dtype=dtype, lazy_decoder=False
        ).transcoder
        lazy_decoder = False
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("  lazy_decoder=False OOMed -> falling back to lazy_decoder=True", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tc = load_transcoder(
            f"qwen3-{MODEL_SIZE}", device=device, dtype=dtype, lazy_decoder=True
        ).transcoder
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated() / 1e9
        print(
            f"  loaded (lazy_decoder={lazy_decoder}); GPU allocated={alloc:.1f}GB, "
            f"free={free / 1e9:.1f}GB / {total / 1e9:.1f}GB",
            flush=True,
        )

    curves = []  # (label, n_ablated, probs, logits)
    for gf in graph_files:
        d = json.loads(gf.read_text())
        prompt = d["prompt"]
        n_bos = d.get("n_bos_tokens", 1)
        answer = d.get("answer_token", "?")
        feats, answer_token_id = _ordered_answer_features(d)
        if not feats or answer_token_id is None:
            print(f"  {gf.name}: no answer features; skipping")
            continue

        messages, _, tkw = prepare_messages(prompt, "qwen3", enable_thinking=False)
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, **tkw
        ).to(device)

        label = f"{prompt.split('?')[0].replace('What is ', '').strip()}={answer}"
        print(f"  {label}: negative-steering up to {len(feats)} features ...", flush=True)
        t0 = time.time()
        res = run_progressive_intervention(
            model, tc, input_ids, feats, answer_token_id, n_bos_tokens=n_bos
        )
        dt = time.time() - t0
        curves.append((label, res.n_ablated, res.probs, res.logits))
        decay = " ".join(f"{p:.2f}" for p in res.probs)
        print(
            f"    {len(feats) + 1} forwards in {dt:.1f}s; p(k=0..{res.n_ablated[-1]}): {decay}",
            flush=True,
        )

    if not curves:
        print("No curves produced.")
        return

    # --- Plot: probability (top) and logit (bottom) vs # features ablated ------
    fig, (ax_p, ax_l) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for label, ks, probs, logits in curves:
        ax_p.plot(ks, probs, marker="o", markersize=3, label=label)
        ax_l.plot(ks, logits, marker="o", markersize=3, label=label)
    ax_p.set_ylabel("p(answer)")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.set_title(f"Progressive negative steering by influence (Qwen3-{MODEL_SIZE})")
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=8, ncol=2)
    ax_l.set_ylabel("logit(answer)")
    ax_l.set_xlabel("# top answer-features negatively steered (cumulative)")
    ax_l.grid(alpha=0.3)
    fig.tight_layout()

    png = out_dir / f"progressive_steering_qwen3-{MODEL_SIZE}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    data = [
        {"label": label, "n_ablated": ks, "probs": probs, "logits": logits}
        for label, ks, probs, logits in curves
    ]
    (out_dir / f"progressive_steering_qwen3-{MODEL_SIZE}.json").write_text(
        json.dumps(data, indent=2)
    )
    print(f"\nSaved curve plot to {png}")


if __name__ == "__main__":
    main()
