#!/usr/bin/env python3
"""Example: prune an attribution graph to remove low-influence nodes/edges.

Loads a previously saved attribution graph JSON, applies graph pruning,
and saves the result.

Usage:
    uv run python examples/prune_attribution_graph.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_circuits.circuits.graph_pruning import prune_graph_dict
from llm_circuits.settings import artifacts_dir

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_FILE = "example_attribution_graph.json"
NODE_THRESHOLD = 0.5
EDGE_THRESHOLD = 0.8
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    graph_dir = artifacts_dir() / "attribution_graphs"
    in_path = graph_dir / INPUT_FILE
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        print("Run examples/build_attribution_graph.py first to generate it.")
        return

    graph_dict = json.loads(in_path.read_text())

    # --- Before-pruning statistics ---
    n_nodes = len(graph_dict["nodes"])
    n_edges = len(graph_dict["edges"])
    type_counts: dict[str, int] = {}
    for nd in graph_dict["nodes"]:
        type_counts[nd["node_type"]] = type_counts.get(nd["node_type"], 0) + 1

    print("Before pruning")
    print(f"  Nodes: {n_nodes}")
    for ntype, count in sorted(type_counts.items()):
        print(f"    {ntype:>12s}: {count}")
    print(f"  Edges: {n_edges}")

    # --- Prune ---
    print(f"\nPruning (node_threshold={NODE_THRESHOLD}, edge_threshold={EDGE_THRESHOLD}) ...")
    pruned = prune_graph_dict(
        graph_dict, node_threshold=NODE_THRESHOLD, edge_threshold=EDGE_THRESHOLD
    )

    # --- After-pruning statistics ---
    pn = len(pruned["nodes"])
    pe = len(pruned["edges"])
    ptype_counts: dict[str, int] = {}
    for nd in pruned["nodes"]:
        ptype_counts[nd["node_type"]] = ptype_counts.get(nd["node_type"], 0) + 1

    print("\nAfter pruning")
    print(f"  Nodes: {pn} ({pn / n_nodes * 100:.1f}% of original)")
    for ntype, count in sorted(ptype_counts.items()):
        print(f"    {ntype:>12s}: {count}")
    print(f"  Edges: {pe} ({pe / n_edges * 100:.1f}% of original)")

    # --- Top nodes by influence ---
    scored = [(nd, nd.get("influence", 0.0)) for nd in pruned["nodes"]]
    scored.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 15 nodes by influence:")
    for nd, score in scored[:15]:
        ntype = nd["node_type"]
        layer = nd["layer"]
        pos = nd["position"]
        extra = ""
        if ntype == "feature":
            extra = f" f={nd['feature_idx']}"
        elif ntype == "logit":
            extra = f" tok={nd['token_id']}"
        print(f"  {score:.6f}  {ntype:>9s} L{layer} p={pos}{extra}")

    # --- Save ---
    out_path = graph_dir / Path(INPUT_FILE).stem
    out_path = graph_dir / f"{Path(INPUT_FILE).stem}_pruned.json"
    out_path.write_text(json.dumps(pruned, indent=2))
    print(f"\nSaved pruned graph to {out_path}")


if __name__ == "__main__":
    main()
