#!/usr/bin/env python3
"""Constrained-patching layer-range sweep (the paper's patch end-layer knob).

For each saved addition graph, build a supernode from the answer-driving features,
negative-steer it, and sweep the **patch end layer** from the last steered layer up
to the final layer via :func:`sweep_patch_end_layer`.  A higher end layer freezes
more of the model (fewer layers recompute); the paper picks the end layer with the
largest logit suppression.  We plot logit/prob vs end layer and mark that argmax.

Note on per-layer transcoders: addition's answer features are concentrated in the
*top* layers, so steering the full supernode leaves no sweep room (``l_max`` = last
layer).  We therefore steer the **mid-layer** answer features (layer <=
``MAX_STEER_LAYER``) so the patch end-layer range is non-degenerate -- this is the
PLT analogue of the paper's factual-recall supernode, which lives mid-network.

Usage:
    uv run python examples/layer_range_sweep.py
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import torch

from llm_circuits.circuits.interventions import negative_steer, sweep_patch_end_layer
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

# -- Config -------------------------------------------------------------------
MODEL_SIZE = "4b"
DTYPE_STR = "bf16"
MAX_STEER_LAYER = 28  # only steer answer features at/below this layer (leaves sweep headroom)
MAX_FEATURES = 12  # cap features per supernode (influence-ranked)
# -----------------------------------------------------------------------------


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


def _mid_answer_features(d: dict):
    """Influence-ranked answer features at layer <= MAX_STEER_LAYER (negative-steered)."""
    ans_idx = _answer_logit_index(d)
    if ans_idx is None:
        return [], None
    nodes = d["nodes"]
    answer_token_id = nodes[ans_idx]["token_id"]
    feats = [n for n in nodes if n["node_type"] == "feature" and n["layer"] <= MAX_STEER_LAYER]
    feats.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    steers = [
        negative_steer(n["layer"], n["feature_idx"], position=n["position"])
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

    # Eager decoders avoid per-forward disk reloads (big speedup); fall back on OOM.
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
    print(f"  loaded (lazy_decoder={lazy_decoder}).", flush=True)

    curves = []  # (label, end_layers, delta_logits, probs, best_end)
    for gf in graph_files:
        d = json.loads(gf.read_text())
        prompt = d["prompt"]
        n_bos = d.get("n_bos_tokens", 1)
        answer = d.get("answer_token", "?")
        feats, answer_token_id = _mid_answer_features(d)
        if not feats or answer_token_id is None:
            print(f"  {gf.name}: no mid-layer answer features; skipping")
            continue

        messages, _, tkw = prepare_messages(prompt, "qwen3", enable_thinking=False)
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, **tkw
        ).to(device)

        label = f"{prompt.split('?')[0].replace('What is ', '').strip()}={answer}"
        l_max = max(iv.layer for iv in feats)
        print(
            f"  {label}: steering {len(feats)} features (l_max={l_max}); "
            f"sweeping patch end layer {l_max}..{len(tc) - 1} ...",
            flush=True,
        )
        t0 = time.time()
        res = sweep_patch_end_layer(
            model, tc, input_ids, feats, answer_token_id, n_bos_tokens=n_bos
        )
        dt = time.time() - t0
        curves.append((label, res.end_layers, res.delta_logits, res.probs, res.best_end_layer))
        print(
            f"    {len(res.end_layers)} forwards in {dt:.1f}s; "
            f"best end layer={res.best_end_layer} "
            f"(min logit {min(res.logits):.2f} vs baseline {res.baseline_logit:.2f})",
            flush=True,
        )

    if not curves:
        print("No sweeps produced.")
        return

    # --- Plot: logit suppression (top) and prob (bottom) vs patch end layer -----
    fig, (ax_l, ax_p) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for label, ends, dlogits, probs, best in curves:
        (line,) = ax_l.plot(ends, dlogits, marker="o", markersize=3, label=label)
        bi = ends.index(best)
        ax_l.scatter([best], [dlogits[bi]], s=80, facecolors="none", edgecolors=line.get_color())
        ax_p.plot(ends, probs, marker="o", markersize=3, label=label, color=line.get_color())
    ax_l.axhline(0, color="#999", lw=0.8)
    ax_l.set_ylabel("delta logit(answer)  [negative = suppression]")
    ax_l.set_title(f"Layer-range sweep: patch end layer (Qwen3-{MODEL_SIZE})")
    ax_l.grid(alpha=0.3)
    ax_l.legend(fontsize=8, ncol=2)
    ax_p.set_ylabel("p(answer)")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.set_xlabel("patch end layer (circle = max-suppression layer)")
    ax_p.grid(alpha=0.3)
    fig.tight_layout()

    png = out_dir / f"layer_range_sweep_qwen3-{MODEL_SIZE}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    data = [
        {
            "label": label,
            "end_layers": ends,
            "delta_logits": dlogits,
            "probs": probs,
            "best_end_layer": best,
        }
        for label, ends, dlogits, probs, best in curves
    ]
    (out_dir / f"layer_range_sweep_qwen3-{MODEL_SIZE}.json").write_text(json.dumps(data, indent=2))
    print(f"\nSaved sweep plot to {png}")


if __name__ == "__main__":
    main()
