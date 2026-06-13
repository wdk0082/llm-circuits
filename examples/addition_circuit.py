#!/usr/bin/env python3
"""Addition circuit: build → prune → identify features → ablate to validate.

This turns the feature-overlap experiment into the biology-paper style analysis
for a single addition prompt on Qwen3:

1. Build an attribution graph for ``a+b=`` and prune it.
2. Identify the transcoder features that most influence the predicted answer
   logit (these are the candidate "addition output" / lookup-table features),
   and show their label top-logits.
3. **Validate** the circuit by ablating those features on the local replacement
   model and measuring how far the answer logit drops — a real causal check, not
   just a correlational graph.

Usage:
    uv run python examples/addition_circuit.py
"""

from __future__ import annotations

import json

import torch

from llm_circuits.circuits.attribution_graph import build_attribution_graph
from llm_circuits.circuits.graph_pruning import graph_to_dict, prune_graph
from llm_circuits.circuits.interventions import (
    FeatureAblation,
    ablation_logit_effect,
    run_feature_ablation,
)
from llm_circuits.circuits.visualization import render_graph_html
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder
from llm_circuits.transcoders.feature_labels import load_feature_labels

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_SIZE = "0.6b"
PROMPT = "4+5="  # single-token answer ("9") keeps the target logit unambiguous
DTYPE_STR = "fp32"  # fp32 for a faithful linearised model; "bf16" on OOM
TOP_K_LOGITS = 5
MAX_FEATURE_TARGETS = 400  # cap feature targets so edge computation stays tractable
MIN_EDGE_WEIGHT = 1e-4  # drop negligible edges at build time (memory / clarity)
NODE_THRESHOLD = 0.7
EDGE_THRESHOLD = 0.9
N_TOP_FEATURES = 8  # how many answer-driving features to report / ablate
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    device = default_device()
    dtype = torch.float32 if DTYPE_STR == "fp32" else torch.bfloat16
    print(f"Device: {device}  Dtype: {dtype}  Prompt: {PROMPT!r}")

    model, tokenizer = load_qwen3(MODEL_SIZE, dtype_str=DTYPE_STR, device_map=device)
    model.eval()
    loaded = load_transcoder(f"qwen3-{MODEL_SIZE}", device=device, dtype=dtype)
    tc = loaded.transcoder

    messages, n_bos_tokens, template_kwargs = prepare_messages(
        PROMPT, "qwen3", enable_thinking=False
    )
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **template_kwargs
    ).to(device)
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    print(f"Tokens ({len(tokens)}): {tokens}")

    # --- What does the model actually predict? --------------------------------
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    answer_token_id = int(logits[-1].argmax().item())
    answer_str = tokenizer.decode(answer_token_id)
    print(f"\nModel predicts next token: {answer_str!r} (id={answer_token_id})")

    # --- 1. Build + prune the attribution graph -------------------------------
    print("\nBuilding attribution graph ...")
    graph = build_attribution_graph(
        model,
        tc,
        input_ids,
        n_bos_tokens=n_bos_tokens,
        top_k_logits=TOP_K_LOGITS,
        max_feature_targets=MAX_FEATURE_TARGETS,
        min_edge_weight=MIN_EDGE_WEIGHT,
    )
    print(f"  raw graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    pruned = prune_graph(graph, node_threshold=NODE_THRESHOLD, edge_threshold=EDGE_THRESHOLD)
    pg = pruned.graph
    print(f"  pruned graph: {len(pg.nodes)} nodes, {len(pg.edges)} edges")

    # --- 2. Identify features driving the answer logit ------------------------
    # The answer logit node = the logit node for the model's predicted token
    # (fall back to the highest-activation logit node if absent).
    logit_idxs = [i for i, n in enumerate(pg.nodes) if n.node_type == "logit"]
    answer_logit_idx = next(
        (i for i in logit_idxs if pg.nodes[i].token_id == answer_token_id),
        max(logit_idxs, key=lambda i: pg.nodes[i].activation) if logit_idxs else None,
    )
    if answer_logit_idx is None:
        print("No logit node survived pruning; aborting feature identification.")
        return

    # Direct feature -> answer-logit edges, ranked by |weight|.
    direct = [
        (e.source, e.weight)
        for e in pg.edges
        if e.target == answer_logit_idx and pg.nodes[e.source].node_type == "feature"
    ]
    direct.sort(key=lambda t: abs(t[1]), reverse=True)
    top = direct[:N_TOP_FEATURES]

    # Load labels for the features we are about to report.
    by_layer: dict[int, list[int]] = {}
    for src_idx, _ in top:
        nd = pg.nodes[src_idx]
        by_layer.setdefault(nd.layer, []).append(nd.feature_idx)
    labels = {
        layer: load_feature_labels(loaded.repo_id, layer, idxs) for layer, idxs in by_layer.items()
    }

    print(f"\nTop {len(top)} features feeding the {answer_str!r} logit:")
    print(f"  {'L':>3} {'feat':>7} {'pos':>4} {'edge_w':>9}  top-logits")
    candidate_ablations: list[FeatureAblation] = []
    for src_idx, w in top:
        nd = pg.nodes[src_idx]
        lab = labels.get(nd.layer, {}).get(nd.feature_idx)
        top_logits = ", ".join(str(t) for t in (lab.top_logits[:6] if lab else []))
        print(f"  {nd.layer:>3} {nd.feature_idx:>7} {nd.position:>4} {w:>9.4f}  {top_logits}")
        candidate_ablations.append(FeatureAblation(nd.layer, nd.feature_idx, position=nd.position))

    # --- 3. Validate by ablation ----------------------------------------------
    print("\nValidating with feature ablation (local replacement model) ...")

    # 3a. Ablate the single most influential feature.
    if candidate_ablations:
        single = run_feature_ablation(
            model, tc, input_ids, [candidate_ablations[0]], n_bos_tokens=n_bos_tokens
        )
        eff = ablation_logit_effect(single, [answer_token_id])[answer_token_id]
        top_feat = candidate_ablations[0]
        print(
            f"  ablate L{top_feat.layer} f{top_feat.feature_idx} (pos {top_feat.position}): "
            f"Δlogit({answer_str!r}) = {eff:+.4f}"
        )

    # 3b. Ablate all identified features together (expected: larger drop).
    joint = run_feature_ablation(
        model, tc, input_ids, candidate_ablations, n_bos_tokens=n_bos_tokens
    )
    joint_eff = ablation_logit_effect(joint, [answer_token_id])[answer_token_id]
    new_top = int(joint.ablated_logits[-1].argmax().item())
    print(
        f"  ablate all {len(candidate_ablations)} features: "
        f"Δlogit({answer_str!r}) = {joint_eff:+.4f}  | new top token = {tokenizer.decode(new_top)!r}"
    )

    # --- Save artifacts -------------------------------------------------------
    out_dir = artifacts_dir() / "addition_circuit"
    out_dir.mkdir(parents=True, exist_ok=True)
    logit_token_strs = {
        str(n.token_id): tokenizer.decode(n.token_id) for n in pg.nodes if n.node_type == "logit"
    }
    graph_dict = graph_to_dict(
        pg,
        influence_scores=pruned.influence_scores,
        prompt=PROMPT,
        model=f"qwen3-{MODEL_SIZE}",
        tokens=tokens,
        n_bos_tokens=n_bos_tokens,
        logit_token_strs=logit_token_strs,
        answer_token=answer_str,
    )
    (out_dir / "addition_graph_pruned.json").write_text(json.dumps(graph_dict, indent=2))
    render_graph_html(graph_dict, out_dir / "addition_graph.html", title=f"Addition: {PROMPT}")
    print(f"\nSaved graph + visualization to {out_dir}")


if __name__ == "__main__":
    main()
