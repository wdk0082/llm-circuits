#!/usr/bin/env python3
"""Render the interactive attribution-graph explorer for one saved addition graph.

Reuses a pruned graph saved by ``addition_circuit.py``, enriches its feature nodes
with max-activating dataset examples (read from the transcoder feature data — CPU
only, no model), and writes a single self-contained HTML reproducing the paper's
viewer: attribution graph + node detail (input/output features, token predictions,
activation examples) + manual node grouping into a collapsible supergraph.

Runs on CPU (saved graph + cached feature files), so it's fine on a login node.

Usage:
    uv run python examples/graph_explorer.py
"""

from __future__ import annotations

import json

from llm_circuits.circuits.graph_explorer import render_graph_explorer_html
from llm_circuits.settings import artifacts_dir
from llm_circuits.transcoders.feature_labels import load_feature_examples
from llm_circuits.transcoders.registry import get_spec

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_SIZE = "4b"
EXAMPLE = "2plus3"  # which saved graph to explore (e.g. 1plus2, 2plus3, 4plus4)
N_EXAMPLES = 3  # activation examples shown per feature
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    out_dir = artifacts_dir() / "addition_circuit"
    gf = out_dir / f"addition_graph_qwen3-{MODEL_SIZE}_{EXAMPLE}.json"
    if not gf.exists():
        cands = sorted(out_dir.glob(f"addition_graph_qwen3-{MODEL_SIZE}_*plus*.json"))
        if not cands:
            print(f"No saved graphs in {out_dir}. Run examples/addition_circuit.py first.")
            return
        gf = cands[0]
    print(f"Loading graph: {gf.name}")
    d = json.loads(gf.read_text())
    repo = get_spec(d["model"]).transcoder_repo

    # Enrich feature nodes with activation examples (CPU; reads cached feature blobs).
    by_layer: dict[int, list[int]] = {}
    for n in d["nodes"]:
        if n["node_type"] == "feature":
            by_layer.setdefault(n["layer"], []).append(n["feature_idx"])
    ex_lookup: dict[tuple[int, int], list] = {}
    for layer, idxs in by_layer.items():
        for fid, exs in load_feature_examples(repo, layer, idxs, n_examples=N_EXAMPLES).items():
            ex_lookup[(layer, fid)] = exs

    n_with = 0
    for n in d["nodes"]:
        if n["node_type"] == "feature":
            lab = n.get("label") or {}
            lab["examples"] = ex_lookup.get((n["layer"], n["feature_idx"]), [])
            n["label"] = lab
            if lab["examples"]:
                n_with += 1

    title = f"Attribution graph: {d.get('prompt', '?')} → {d.get('answer_token', '?')!r}"
    out = render_graph_explorer_html(
        d, out_dir / f"graph_explorer_qwen3-{MODEL_SIZE}.html", title=title
    )
    nfeat = sum(1 for n in d["nodes"] if n["node_type"] == "feature")
    print(f"Feature nodes: {nfeat} ({n_with} with activation examples)")
    print(f"Saved explorer to {out}")


if __name__ == "__main__":
    main()
