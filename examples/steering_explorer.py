#!/usr/bin/env python3
"""Pre-compute an interactive steering explorer for the addition circuits.

For each saved attribution graph, take the answer "supernode" (the top influence
features feeding the answer logit) and sweep a single **steering multiple M** over
the whole supernode (additive-delta convention: M=0 no change, -1 ablate, -2
negative steer / flip), recording the top tokens' probabilities at each M via the
faithful **steering-base-model** intervention.  The results are baked into a
self-contained HTML with a dropdown (example) + slider (M) + live bar chart.

Usage:
    uv run python examples/steering_explorer.py
"""

from __future__ import annotations

import json

import torch

from llm_circuits.circuits.interventions import FeatureIntervention, run_feature_intervention
from llm_circuits.circuits.visualization import render_steering_explorer_html
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_SIZE = "4b"
DTYPE_STR = "bf16"
N_FEATURES = 8  # supernode size: top-N influence features feeding the answer
# Additive-delta M sweep: M=0 no change, -1 ablate, -2 negative steer (flip).
FACTORS = [0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
N_TOKENS = 8  # tracked tokens per factor (union across factors is shown)
MAX_TRACKED = 12
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


def _supernode_features(d: dict) -> tuple[list[tuple[int, int, int]], int | None]:
    """Top-N_FEATURES influence features feeding the answer + the answer token id."""
    ans_idx = _answer_logit_index(d)
    if ans_idx is None:
        return [], None
    nodes = d["nodes"]
    feats = [n for n in nodes if n["node_type"] == "feature"]
    feats.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    chosen = [(n["layer"], n["feature_idx"], n["position"]) for n in feats[:N_FEATURES]]
    return chosen, nodes[ans_idx]["token_id"]


def main() -> None:
    device = default_device()
    dtype = torch.float32 if DTYPE_STR == "fp32" else torch.bfloat16
    out_dir = artifacts_dir() / "addition_circuit"

    graph_files = sorted(out_dir.glob(f"addition_graph_qwen3-{MODEL_SIZE}_*plus*.json"))
    if not graph_files:
        print(f"No per-example graphs in {out_dir}. Run examples/addition_circuit.py first.")
        return

    print(f"Loading Qwen3-{MODEL_SIZE} ({DTYPE_STR}) ...")
    model, tokenizer = load_qwen3(MODEL_SIZE, dtype_str=DTYPE_STR, device_map=device)
    model.eval()
    try:
        tc = load_transcoder(
            f"qwen3-{MODEL_SIZE}", device=device, dtype=dtype, lazy_decoder=False
        ).transcoder
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tc = load_transcoder(
            f"qwen3-{MODEL_SIZE}", device=device, dtype=dtype, lazy_decoder=True
        ).transcoder

    examples = []
    for gf in graph_files:
        d = json.loads(gf.read_text())
        prompt = d["prompt"]
        n_bos = d.get("n_bos_tokens", 1)
        feats, answer_id = _supernode_features(d)
        if not feats or answer_id is None:
            continue
        answer_str = tokenizer.decode(answer_id)

        messages, _, tkw = prepare_messages(prompt, "qwen3", enable_thinking=False)
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, **tkw
        ).to(device)

        factor_probs: dict[float, torch.Tensor] = {}
        tracked: set[int] = {int(answer_id)}
        for m in FACTORS:
            steers = [
                FeatureIntervention(layer, fid, position=pos, m=m) for layer, fid, pos in feats
            ]
            res = run_feature_intervention(model, tc, input_ids, steers, n_bos_tokens=n_bos)
            probv = res.ablated_logits[-1].float().softmax(dim=-1).cpu()
            factor_probs[m] = probv
            tracked.update(probv.topk(N_TOKENS).indices.tolist())

        tracked_ids = sorted(
            tracked, key=lambda t: max(factor_probs[m][t].item() for m in FACTORS), reverse=True
        )[:MAX_TRACKED]
        tokens = [tokenizer.decode(t) for t in tracked_ids]
        probs = [[factor_probs[m][t].item() for t in tracked_ids] for m in FACTORS]
        label = f"{prompt.split('?')[0].replace('What is ', '').strip()}={answer_str}"
        examples.append(
            {
                "label": label,
                "answer": answer_str,
                "factors": FACTORS,
                "tokens": tokens,
                "probs": probs,
            }
        )
        m_flip = probs[FACTORS.index(-2.0)]
        flip_top = tokens[m_flip.index(max(m_flip))]
        print(
            f"  {label}: clean top={answer_str!r}  -> at M=-2 (flip) top={flip_top!r}", flush=True
        )

    if not examples:
        print("No examples produced.")
        return

    data = {"model": f"qwen3-{MODEL_SIZE}", "examples": examples}
    (out_dir / f"steering_explorer_qwen3-{MODEL_SIZE}.json").write_text(json.dumps(data, indent=2))
    out = render_steering_explorer_html(
        data,
        out_dir / f"steering_explorer_qwen3-{MODEL_SIZE}.html",
        title=f"Addition steering explorer — Qwen3-{MODEL_SIZE}",
    )
    print(f"\nSaved steering explorer to {out}")


if __name__ == "__main__":
    main()
