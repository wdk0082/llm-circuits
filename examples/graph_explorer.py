#!/usr/bin/env python3
"""Render the interactive attribution-graph explorer for all saved addition graphs.

Reuses the pruned graphs saved by ``addition_circuit.py``, enriches their feature
nodes with max-activating dataset examples (read from the transcoder feature data --
CPU only, no model), and writes a single self-contained HTML reproducing the paper's
viewer: attribution graph + node detail (input/output features, token predictions,
activation examples) + manual node grouping into a collapsible supergraph, with an
**example dropdown** to switch between graphs.

Runs on CPU (saved graphs + cached feature files), so it's fine on a login node.

Usage:
    uv run python examples/graph_explorer.py
"""

from __future__ import annotations

import json

from llm_circuits.circuits.graph_explorer import render_graph_explorer_html
from llm_circuits.settings import artifacts_dir
from llm_circuits.transcoders.feature_labels import load_feature_examples
from llm_circuits.transcoders.registry import get_spec

# -- Config -------------------------------------------------------------------
MODEL_SIZE = "4b"
N_PER_QUANTILE = 5  # activation examples shown per quantile (Top, subsamples, Bottom)
WINDOW = 10  # tokens kept on each side of the peak-activation token
# -----------------------------------------------------------------------------


def _enrich(d: dict, repo: str) -> int:
    """Attach quantile-grouped activation examples to feature nodes; return count."""
    by_layer: dict[int, list[int]] = {}
    for n in d["nodes"]:
        if n["node_type"] == "feature":
            by_layer.setdefault(n["layer"], []).append(n["feature_idx"])
    ex_lookup: dict[tuple[int, int], list] = {}
    for layer, idxs in by_layer.items():
        loaded = load_feature_examples(
            repo, layer, idxs, n_per_quantile=N_PER_QUANTILE, window=WINDOW
        )
        for fid, exs in loaded.items():
            ex_lookup[(layer, fid)] = exs

    n_with = 0
    for n in d["nodes"]:
        if n["node_type"] == "feature":
            lab = n.get("label") or {}
            lab["examples"] = ex_lookup.get((n["layer"], n["feature_idx"]), [])
            n["label"] = lab
            if lab["examples"]:
                n_with += 1
    return n_with


def main() -> None:
    out_dir = artifacts_dir() / "addition_circuit"
    graph_files = sorted(out_dir.glob(f"addition_graph_qwen3-{MODEL_SIZE}_*plus*.json"))
    if not graph_files:
        print(f"No saved graphs in {out_dir}. Run examples/addition_circuit.py first.")
        return

    graphs: list[dict] = []
    for gf in graph_files:
        d = json.loads(gf.read_text())
        repo = get_spec(d["model"]).transcoder_repo
        n_with = _enrich(d, repo)
        nfeat = sum(1 for n in d["nodes"] if n["node_type"] == "feature")
        print(f"  {gf.name}: {nfeat} feature nodes ({n_with} with activation examples)")
        graphs.append(d)

    title = f"Addition attribution graphs (Qwen3-{MODEL_SIZE})"
    out = render_graph_explorer_html(
        graphs, out_dir / f"graph_explorer_qwen3-{MODEL_SIZE}.html", title=title
    )
    print(f"Loaded {len(graphs)} graphs; saved explorer to {out}")


if __name__ == "__main__":
    main()
