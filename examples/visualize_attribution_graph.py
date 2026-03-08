#!/usr/bin/env python3
"""Example: visualize a pruned attribution graph as an interactive HTML file.

Usage:
    uv run python examples/visualize_attribution_graph.py
"""

from __future__ import annotations

import json

from llm_circuits.circuits.visualization import render_graph_html
from llm_circuits.settings import artifacts_dir


def main() -> None:
    graph_dir = artifacts_dir() / "attribution_graphs"

    # Prefer pruned graph; fall back to unpruned
    pruned_path = graph_dir / "example_attribution_graph_pruned.json"
    unpruned_path = graph_dir / "example_attribution_graph.json"

    if pruned_path.exists():
        in_path = pruned_path
    elif unpruned_path.exists():
        in_path = unpruned_path
        print("Note: using unpruned graph. Run prune_attribution_graph.py first for best results.")
    else:
        print(f"No graph found in {graph_dir}")
        print("Run examples/build_attribution_graph.py first.")
        return

    graph_dict = json.loads(in_path.read_text())
    n_nodes = len(graph_dict.get("nodes", []))
    n_edges = len(graph_dict.get("edges", []))
    print(f"Loaded {in_path.name}: {n_nodes} nodes, {n_edges} edges")

    out_path = graph_dir / "example_attribution_graph.html"
    render_graph_html(graph_dict, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
