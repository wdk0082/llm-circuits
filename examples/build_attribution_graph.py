#!/usr/bin/env python3
"""Example: build and inspect an attribution graph.

Loads Qwen3-0.6B and its transcoders, builds an attribution graph for a
short prompt, prints summary statistics, and saves the graph to disk.

Usage:
    uv run python examples/build_attribution_graph.py
"""

from __future__ import annotations

import torch

from llm_circuits.circuits.attribution_graph import (
    AttributionGraph,
    NodeType,
    build_attribution_graph,
)
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device, default_dtype
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SIZE = "0.6b"
PROMPT = "Answer immediately with one word: The capital of France is?"
FEATURE_TO_FEATURE = True  # Include feature-to-feature edges (slower but more complete)
SAVE_PATH = artifacts_dir() / "attribution_graph.pt"
# ──────────────────────────────────────────────────────────────────────────────


def _print_graph_summary(graph: AttributionGraph) -> None:
    """Print high-level statistics about the graph."""
    by_type: dict[NodeType, int] = {}
    for nid in graph.nodes:
        by_type[nid.node_type] = by_type.get(nid.node_type, 0) + 1

    print(f"  Total nodes:  {len(graph.nodes)}")
    for nt in NodeType:
        print(f"    {nt.value:>12s}:  {by_type.get(nt, 0)}")
    print(f"  Total edges:  {len(graph.edges)}")


def _print_top_edges(graph: AttributionGraph, *, n: int = 20) -> None:
    """Print the top-n edges by absolute weight."""
    sorted_edges = sorted(graph.edges, key=lambda e: abs(e[2]), reverse=True)
    print(f"\n  Top-{n} edges by |weight|:")
    print(
        f"  {'Src type':>12}  {'Src (L,P,F)':>16}  {'Tgt type':>12}  {'Tgt (L,P,F)':>16}  {'Weight':>12}"
    )
    print("  " + "-" * 72)
    for src, tgt, w in sorted_edges[:n]:
        src_label = f"({src.layer},{src.position},{src.feature_idx})"
        tgt_label = f"({tgt.layer},{tgt.position},{tgt.feature_idx})"
        print(
            f"  {src.node_type.value:>12}  {src_label:>16}  "
            f"{tgt.node_type.value:>12}  {tgt_label:>16}  {w:12.4f}"
        )


def _print_top_feature_nodes(graph: AttributionGraph, *, n: int = 20) -> None:
    """Print the top-n feature nodes by activation magnitude."""
    features = [node for node in graph.nodes.values() if node.id.node_type == NodeType.FEATURE]
    features.sort(key=lambda node: abs(node.activation), reverse=True)
    print(f"\n  Top-{n} feature nodes by |activation|:")
    print(f"  {'Layer':>5}  {'Pos':>4}  {'Feat':>8}  {'Token':>16}  {'Activation':>12}")
    print("  " + "-" * 52)
    for node in features[:n]:
        tok = graph.tokens[node.id.position] if node.id.position < len(graph.tokens) else "?"
        tok_d = repr(tok)[1:-1]
        print(
            f"  {node.id.layer:5d}  {node.id.position:4d}  {node.id.feature_idx:8d}"
            f"  {tok_d:>16}  {node.activation:12.4f}"
        )


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

    # --- Tokenize -------------------------------------------------------------
    messages, n_bos_tokens = prepare_messages(PROMPT, "qwen3")
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    print(f"\nPrompt: {PROMPT!r}")
    print(f"Tokens ({len(tokens)}): {tokens}\n")

    # --- Build attribution graph ----------------------------------------------
    edge_desc = "feature-to-feature + logit" if FEATURE_TO_FEATURE else "logit edges only"
    print(f"Building attribution graph ({edge_desc}) ...")
    graph = build_attribution_graph(
        model,
        tc,
        input_ids,
        tokenizer,
        n_bos_tokens=n_bos_tokens,
        feature_to_feature=FEATURE_TO_FEATURE,
        model_name=f"qwen3-{MODEL_SIZE}",
        transcoder_repo=loaded.repo_id,
    )

    # --- Print summary --------------------------------------------------------
    print("\n" + "=" * 80)
    print("Attribution graph summary")
    print("=" * 80)
    _print_graph_summary(graph)
    _print_top_feature_nodes(graph)
    _print_top_edges(graph)

    # --- Save -----------------------------------------------------------------
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph.save(SAVE_PATH)
    print(f"\nGraph saved to {SAVE_PATH}")

    # --- Verify round-trip ----------------------------------------------------
    loaded_graph = AttributionGraph.load(SAVE_PATH)
    assert len(loaded_graph.nodes) == len(graph.nodes), "Node count mismatch after reload"
    assert len(loaded_graph.edges) == len(graph.edges), "Edge count mismatch after reload"
    print("Save/load round-trip: PASS")


if __name__ == "__main__":
    main()
