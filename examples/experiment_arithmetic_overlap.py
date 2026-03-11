#!/usr/bin/env python3
"""Experiment: transcoder feature overlap across arithmetic operands.

For each (a, b) pair and operation (+, *), runs a local replacement forward
pass and extracts active transcoder features per (layer, position).  Produces
a pair-vs-pair mean-Jaccard similarity matrix for each operation, showing how
much the model reuses features across different operands.

Usage:
    uv run python examples/experiment_addition_overlap.py
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from llm_circuits.circuits.local_replacement_model import run_local_replacement
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device, default_dtype
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_SIZE = "0.6b"

ADDITION_PAIRS: list[tuple[int, int]] = [
    (1, 2),  # small + small
    (3, 4),  # small + small
    (7, 8),  # large + large
    (2, 8),  # mixed
    (1, 9),  # extremes
]

MULTIPLICATION_PAIRS: list[tuple[int, int]] = [
    (1, 2),  # small * small
    (3, 4),  # small * small
    (7, 8),  # large * large
    (2, 8),  # mixed
    (1, 9),  # extremes
]

OPERATIONS: list[dict] = [
    {"name": "addition", "template": "{a}+{b}=?", "symbol": "+", "pairs": ADDITION_PAIRS},
    {
        "name": "multiplication",
        "template": "{a}*{b}=?",
        "symbol": "*",
        "pairs": MULTIPLICATION_PAIRS,
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def extract_active_features(
    local_ctx,
    n_bos_tokens: int,
) -> dict[int, list[set[int]]]:
    """Extract active feature index sets per (layer, position).

    Returns ``{layer_idx: [set_of_active_feature_indices per position]}``.
    Positions before *n_bos_tokens* are included but will be empty sets
    since transcoders are not applied there.
    """
    result: dict[int, list[set[int]]] = {}
    for layer_idx, feat_tensor in sorted(local_ctx.features.items()):
        if feat_tensor.dim() == 3:
            feat_tensor = feat_tensor[0]
        n_pos = feat_tensor.shape[0]
        per_pos: list[set[int]] = []
        for pos in range(n_pos):
            if pos < n_bos_tokens:
                per_pos.append(set())
            else:
                active = feat_tensor[pos].nonzero(as_tuple=True)[0].tolist()
                per_pos.append(set(active))
        result[layer_idx] = per_pos
    return result


def compute_jaccard_matrix(
    feats_a: dict[int, list[set[int]]],
    feats_b: dict[int, list[set[int]]],
) -> np.ndarray:
    """Compute (n_layers, n_positions) Jaccard similarity matrix."""
    layers = sorted(feats_a.keys())
    n_layers = len(layers)
    n_positions = len(feats_a[layers[0]])
    matrix = np.zeros((n_layers, n_positions))
    for li, layer in enumerate(layers):
        for pos in range(n_positions):
            sa = feats_a[layer][pos]
            sb = feats_b[layer][pos]
            union = sa | sb
            if len(union) == 0:
                matrix[li, pos] = 1.0
            else:
                matrix[li, pos] = len(sa & sb) / len(union)
    return matrix


def compute_mean_jaccard(
    feats_a: dict[int, list[set[int]]],
    feats_b: dict[int, list[set[int]]],
) -> float:
    """Compute mean Jaccard similarity across all (layer, position) cells."""
    mat = compute_jaccard_matrix(feats_a, feats_b)
    return float(mat.mean())


def _make_tick_labels(tokens_a: list[str], tokens_b: list[str]) -> list[str]:
    """Build x-tick labels, showing 'a/c' at positions where tokens differ."""
    labels: list[str] = []
    for ta, tb in zip(tokens_a, tokens_b, strict=True):
        sa = repr(ta)[1:-1][:8]
        sb = repr(tb)[1:-1][:8]
        if sa != sb:
            labels.append(f"{sa}/{sb}")
        else:
            labels.append(sa)
    return labels


def plot_similarity_matrix(
    sim_matrix: np.ndarray,
    pair_labels: list[str],
    title: str,
    save_path,
) -> None:
    """Plot a pair-vs-pair mean-Jaccard similarity matrix."""
    n = len(pair_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(pair_labels, fontsize=10)

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Mean Jaccard similarity", shrink=0.8)
    ax.set_title(title, fontsize=13)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {save_path}")


def plot_pairwise_heatmaps(
    jaccard_matrices: dict[tuple[int, int], np.ndarray],
    pairs: list[tuple[int, int]],
    symbol: str,
    all_tokens: list[list[str]],
    title: str,
    save_path,
) -> None:
    """Plot per-pair (layer x position) Jaccard heatmaps in a grid."""
    comparisons = list(itertools.combinations(range(len(pairs)), 2))
    n = len(comparisons)
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False, sharey=True
    )

    im = None
    for idx, (i, j) in enumerate(comparisons):
        ax = axes[idx // n_cols][idx % n_cols]
        mat = jaccard_matrices[(i, j)]
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        pa, pb = pairs[i], pairs[j]
        ax.set_title(f"{pa[0]}{symbol}{pa[1]} vs {pb[0]}{symbol}{pb[1]}", fontsize=9)
        ax.set_xlabel("Position", fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel("Layer")
        tick_labels = _make_tick_labels(all_tokens[i], all_tokens[j])
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Jaccard similarity", shrink=0.6)
    fig.suptitle(title, fontsize=13)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {save_path}")


def run_operation(
    op: dict,
    model,
    tokenizer,
    tc,
    device,
    out_dir,
) -> None:
    """Run feature extraction and comparison for one arithmetic operation."""
    name = op["name"]
    template = op["template"]
    symbol = op["symbol"]
    pairs = op["pairs"]

    print(f"\n{'=' * 70}")
    print(f"Operation: {name} ({template})")
    print("=" * 70)

    all_features: list[dict[int, list[set[int]]]] = []
    all_tokens: list[list[str]] = []

    for a, b in pairs:
        prompt = template.format(a=a, b=b)
        print(f"\n  Processing ({a}, {b}): prompt={prompt!r}")

        messages, n_bos_tokens, template_kwargs = prepare_messages(
            prompt, "qwen3", enable_thinking=False
        )
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            **template_kwargs,
        ).to(device)
        tokens = [tokenizer.decode(t) for t in input_ids[0]]
        print(f"    Tokens ({len(tokens)}): {tokens}")

        local_ctx = run_local_replacement(
            model, tc, input_ids, include_error=True, n_bos_tokens=n_bos_tokens
        )

        feats = extract_active_features(local_ctx, n_bos_tokens)
        all_features.append(feats)
        all_tokens.append(tokens)

        layers = sorted(feats.keys())
        total_active = sum(len(s) for layer in layers for s in feats[layer])
        print(f"    Active features: {total_active}")

    # --- Compute NxN mean-Jaccard matrix and per-pair Jaccard matrices --------
    n = len(pairs)
    sim_matrix = np.ones((n, n))
    jaccard_matrices: dict[tuple[int, int], np.ndarray] = {}
    for i, j in itertools.combinations(range(n), 2):
        mat = compute_jaccard_matrix(all_features[i], all_features[j])
        jaccard_matrices[(i, j)] = mat
        mj = float(mat.mean())
        sim_matrix[i, j] = mj
        sim_matrix[j, i] = mj

    # --- Print summary --------------------------------------------------------
    pair_labels = [f"{a}{symbol}{b}" for a, b in pairs]
    print(f"\n  Mean Jaccard similarity matrix ({name}):")
    header = "          " + "  ".join(f"{lb:>6s}" for lb in pair_labels)
    print(header)
    for i, lb in enumerate(pair_labels):
        row = f"  {lb:>6s}  " + "  ".join(f"{sim_matrix[i, j]:6.3f}" for j in range(n))
        print(row)

    # --- Plot similarity matrix -----------------------------------------------
    sim_title = f"Feature Overlap: {name.title()} ({template})"
    sim_path = out_dir / f"{name}_overlap.png"
    plot_similarity_matrix(sim_matrix, pair_labels, sim_title, sim_path)

    # --- Plot pairwise heatmaps -----------------------------------------------
    pw_title = f"Per-Layer Feature Overlap: {name.title()} ({template})"
    pw_path = out_dir / f"{name}_pairwise.png"
    plot_pairwise_heatmaps(jaccard_matrices, pairs, symbol, all_tokens, pw_title, pw_path)


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

    # --- Run each operation ---------------------------------------------------
    out_dir = artifacts_dir() / "arithmetic_overlap"
    for op in OPERATIONS:
        run_operation(op, model, tokenizer, tc, device, out_dir)

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
