#!/usr/bin/env python3
"""Example: build an attribution graph for a short prompt.

Loads Qwen3-0.6B and its transcoders, runs the attribution graph builder,
then prints summary statistics and saves the graph as a JSON file.

Usage:
    uv run python examples/build_attribution_graph.py
"""

from __future__ import annotations

import json

import torch

from llm_circuits.circuits.attribution_graph import build_attribution_graph
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device, default_dtype
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_SIZE = "0.6b"
PROMPT = "The capital of France is"
TOP_K_LOGITS = 3
MAX_FEATURE_TARGETS = 100  # cap feature targets for tractable edge computation
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    device = default_device()
    dtype = default_dtype()
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"

    print(f"Device: {device}  Dtype: {dtype}")

    # --- Load model -----------------------------------------------------------
    print(f"Loading Qwen3-{MODEL_SIZE.upper()} ({dtype_str}) ...")
    model, tokenizer = load_qwen3(MODEL_SIZE, dtype_str=dtype_str, device_map=device)
    model.eval()

    # --- Load transcoders -----------------------------------------------------
    print("Loading transcoders ...")
    loaded = load_transcoder(f"qwen3-{MODEL_SIZE}", device=device, dtype=dtype)
    tc = loaded.transcoder
    print(f"  Type: {type(tc).__name__}  Repo: {loaded.repo_id}")

    # --- Tokenize via chat template -------------------------------------------
    messages, n_bos_tokens = prepare_messages(PROMPT, "qwen3")
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    print(f"\nPrompt: {PROMPT!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"BOS tokens: {n_bos_tokens}\n")

    # --- Build attribution graph ----------------------------------------------
    print("Building attribution graph ...")
    graph = build_attribution_graph(
        model,
        tc,
        input_ids,
        n_bos_tokens=n_bos_tokens,
        top_k_logits=TOP_K_LOGITS,
        max_feature_targets=MAX_FEATURE_TARGETS,
    )

    # --- Summary statistics ---------------------------------------------------
    print("\n" + "=" * 70)
    print("Attribution Graph Summary")
    print("=" * 70)

    type_counts: dict[str, int] = {}
    for node in graph.nodes:
        type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

    print(f"  Total nodes: {len(graph.nodes)}")
    for ntype, count in sorted(type_counts.items()):
        print(f"    {ntype:>12s}: {count}")
    print(f"  Total edges: {len(graph.edges)}")

    if graph.edges:
        raw_weights = [e.weight for e in graph.edges]
        abs_weights = [abs(w) for w in raw_weights]
        mean_w = sum(abs_weights) / len(abs_weights)
        max_w = max(abs_weights)
        n_pos = sum(1 for w in raw_weights if w > 0)
        n_neg = sum(1 for w in raw_weights if w < 0)
        print(f"  Non-zero edges: {len(raw_weights)}")
        print(f"  Mean |weight|:  {mean_w:.6f}")
        print(f"  Max  |weight|:  {max_w:.6f}")
        print(f"  Positive edges: {n_pos}  ({n_pos / len(raw_weights) * 100:.1f}%)")
        print(f"  Negative edges: {n_neg}  ({n_neg / len(raw_weights) * 100:.1f}%)")

    # --- Print top edges ------------------------------------------------------
    print("\n" + "=" * 70)
    print("Top 20 edges by |weight|")
    print("=" * 70)

    sorted_edges = sorted(graph.edges, key=lambda e: abs(e.weight), reverse=True)[:20]
    print(f"  {'Source':>40s}  -->  {'Target':>40s}  {'Weight':>12s}")
    print("  " + "-" * 100)
    for edge in sorted_edges:
        src = graph.nodes[edge.source]
        tgt = graph.nodes[edge.target]
        src_str = _node_str(src, tokenizer, input_ids[0])
        tgt_str = _node_str(tgt, tokenizer, input_ids[0])
        print(f"  {src_str:>40s}  -->  {tgt_str:>40s}  {edge.weight:>12.4f}")

    # --- Print logit nodes ----------------------------------------------------
    print("\n" + "=" * 70)
    print("Logit nodes (top-k predictions)")
    print("=" * 70)

    for node in graph.nodes:
        if node.node_type == "logit":
            tok_str = tokenizer.decode(node.token_id)
            print(f"  token={tok_str!r}  (id={node.token_id})  logit={node.activation:.4f}")

    # --- Save to JSON ---------------------------------------------------------
    out_dir = artifacts_dir() / "attribution_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "example_attribution_graph.json"

    graph_dict = {
        "prompt": PROMPT,
        "model": f"qwen3-{MODEL_SIZE}",
        "tokens": tokens,
        "n_bos_tokens": n_bos_tokens,
        "top_k_logits": TOP_K_LOGITS,
        "nodes": [
            {
                "node_type": n.node_type,
                "layer": n.layer,
                "position": n.position,
                "feature_idx": n.feature_idx,
                "token_id": n.token_id,
                "activation": n.activation,
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "weight": e.weight,
            }
            for e in graph.edges
        ],
    }

    out_path.write_text(json.dumps(graph_dict, indent=2))
    print(f"\nSaved attribution graph to {out_path}")
    print(f"  ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")


def _node_str(node, tokenizer, input_ids_1d) -> str:
    """Format a node as a short string for display."""
    tok = tokenizer.decode(input_ids_1d[node.position].item())
    tok_repr = repr(tok)[1:-1]
    if node.node_type == "embedding":
        return f"emb(p={node.position}, '{tok_repr}')"
    if node.node_type == "feature":
        return f"feat(L{node.layer}, p={node.position}, f={node.feature_idx})"
    if node.node_type == "error":
        return f"err(L{node.layer}, p={node.position})"
    if node.node_type == "logit":
        pred = tokenizer.decode(node.token_id)
        return f"logit(p={node.position}, '{pred}')"
    return f"{node.node_type}(?)"


if __name__ == "__main__":
    main()
