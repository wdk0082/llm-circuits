#!/usr/bin/env python3
"""Analyze how Qwen3-0.6B internally represents simple addition (a+b=c).

Examines:
  1. Residual-stream / MLP / attention activations across layers as a,b vary
  2. Gradient norms of the correct-answer logit w.r.t. model weights by layer/component

Saves 6 PNG plots and a results.pt file to artifacts/addition_representations/.

Usage:
    uv run python examples/addition_representations.py
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.instrumentation.hooks import ActivationRecorder
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device, default_dtype
from llm_circuits.utils.seeds import seed_everything

matplotlib.use("Agg")  # non-interactive backend for HPC

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SIZE = "0.6b"
FAMILY = "qwen3"
A_RANGE = range(1, 10)
B_RANGE = range(1, 10)
PROMPT_TEMPLATE = "What is {a}+{b}?"
ANALYSIS_LAYERS = [0, 6, 13, 20, 27]  # subset for detailed cosine-sim / PCA
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────


def categorize_parameters(
    model: torch.nn.Module, n_layers: int
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Group named parameters by (layer_idx, component) or global bucket."""
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is not None:
            if "self_attn" in name:
                comp = "self_attn"
            elif "mlp" in name:
                comp = "mlp"
            elif "input_layernorm" in name or "post_attention_layernorm" in name:
                comp = "layernorm"
            else:
                comp = "other"
            key = f"layer.{layer_idx}.{comp}"
        elif "embed_tokens" in name:
            key = "embed_tokens"
        elif "norm" in name and "layer" not in name:
            key = "final_norm"
        elif "lm_head" in name:
            key = "lm_head"
        else:
            key = "other_global"
        groups[key].append((name, param))
    return dict(groups)


def compute_cosine_sim_matrix(vecs: Tensor) -> Tensor:
    """Pairwise cosine similarity for (N, D) matrix."""
    vecs_f = vecs.float()
    norms = vecs_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = vecs_f / norms
    return normed @ normed.T


def pca_reduce(vecs: Tensor, n_components: int = 3) -> Tensor:
    """Project (N, D) to (N, n_components) via torch.pca_lowrank."""
    vecs_f = vecs.float()
    vecs_centered = vecs_f - vecs_f.mean(dim=0, keepdim=True)
    U, S, _V = torch.pca_lowrank(vecs_centered, q=n_components)
    return U[:, :n_components] * S[:n_components].unsqueeze(0)


def grad_norm_for_group(
    params: list[tuple[str, torch.nn.Parameter]],
) -> float:
    """Sum of L2 gradient norms (in float32) for a parameter group."""
    total = 0.0
    for _name, p in params:
        if p.grad is not None:
            total += p.grad.float().norm().item()
    return total


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_activation_norms(all_activations: dict, pairs: list, n_layers: int, out_dir: str) -> None:
    """Line plot: layer vs mean activation norm for residual/MLP/attention."""
    layers = list(range(n_layers))
    res_means, mlp_means, attn_means = [], [], []
    for layer in layers:
        res_vals, mlp_vals, attn_vals = [], [], []
        for acts in all_activations.values():
            res_vals.append(acts[f"model.layers.{layer}"].float().norm().item())
            mlp_vals.append(acts[f"model.layers.{layer}.mlp"].float().norm().item())
            attn_vals.append(acts[f"model.layers.{layer}.self_attn"].float().norm().item())
        res_means.append(np.mean(res_vals))
        mlp_means.append(np.mean(mlp_vals))
        attn_means.append(np.mean(attn_vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, res_means, "o-", label="Residual stream", linewidth=2, markersize=4)
    ax.plot(layers, mlp_means, "s-", label="MLP", linewidth=2, markersize=4)
    ax.plot(layers, attn_means, "^-", label="Attention", linewidth=2, markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean activation norm (L2)")
    ax.set_title("Activation Norms Across Layers (mean over 81 addition pairs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "activation_norms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved activation_norms.png")


def plot_prediction_grid(all_predictions: dict, out_dir: str) -> None:
    """9x9 heatmap showing prediction correctness."""
    a_vals = sorted({a for a, _ in all_predictions})
    b_vals = sorted({b for _, b in all_predictions})
    grid = np.zeros((len(b_vals), len(a_vals)))

    for (a, b), pred in all_predictions.items():
        ai = a_vals.index(a)
        bi = b_vals.index(b)
        if pred["is_correct"]:
            grid[bi, ai] = 2  # correct
        elif pred["correct_in_top5"]:
            grid[bi, ai] = 1  # in top-5
        else:
            grid[bi, ai] = 0  # wrong

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#e74c3c", "#f39c12", "#2ecc71"])  # red, yellow, green

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, origin="lower", aspect="equal")

    # Annotate each cell with predicted string
    for (a, b), pred in all_predictions.items():
        ai = a_vals.index(a)
        bi = b_vals.index(b)
        label = pred["pred_str"][:4]
        color = "white" if grid[bi, ai] == 0 else "black"
        ax.text(ai, bi, label, ha="center", va="center", fontsize=6, color=color)

    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels(a_vals)
    ax.set_yticks(range(len(b_vals)))
    ax.set_yticklabels(b_vals)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title("Prediction Grid (green=correct, yellow=top-5, red=wrong)")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Correct (top-1)"),
        Patch(facecolor="#f39c12", label="In top-5"),
        Patch(facecolor="#e74c3c", label="Wrong"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    fig.savefig(out_dir / "prediction_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved prediction_grid.png")


def plot_cosine_similarity(
    all_activations: dict,
    pairs: list,
    n_pairs: int,
    analysis_layers: list,
    n_layers: int,
    out_dir: str,
) -> None:
    """Grouped bar chart: same-sum vs diff-sum cosine similarity per layer."""
    sums = [a + b for a, b in pairs]
    layer_labels, same_means, diff_means = [], [], []

    for layer in analysis_layers:
        if layer >= n_layers:
            continue
        res_key = f"model.layers.{layer}"
        vecs = torch.stack([all_activations[(a, b)][res_key] for a, b in pairs])
        cos_sim = compute_cosine_sim_matrix(vecs)

        same_sims, diff_sims = [], []
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                val = cos_sim[i, j].item()
                if sums[i] == sums[j]:
                    same_sims.append(val)
                else:
                    diff_sims.append(val)

        layer_labels.append(f"L{layer}")
        same_means.append(np.mean(same_sims) if same_sims else 0)
        diff_means.append(np.mean(diff_sims) if diff_sims else 0)

    x = np.arange(len(layer_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, same_means, width, label="Same sum", color="#3498db")
    ax.bar(x + width / 2, diff_means, width, label="Different sum", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Cosine Similarity: Same-Sum vs Different-Sum Pairs (residual stream)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Set y-axis to show differences clearly
    all_vals = same_means + diff_means
    if all_vals:
        ymin = min(all_vals) - 0.002
        ymax = max(all_vals) + 0.002
        ax.set_ylim(max(0, ymin), min(1.005, ymax))

    fig.savefig(out_dir / "cosine_similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cosine_similarity.png")


def plot_pca_scatter(
    all_activations: dict,
    pairs: list,
    analysis_layers: list,
    n_layers: int,
    out_dir: str,
) -> None:
    """Scatter plots of first two PCA components, colored by sum value."""
    valid_layers = [ly for ly in analysis_layers if ly < n_layers]
    n_plots = len(valid_layers)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    sums = np.array([a + b for a, b in pairs])
    cmap = plt.cm.viridis

    for ax, layer in zip(axes, valid_layers, strict=True):
        res_key = f"model.layers.{layer}"
        vecs = torch.stack([all_activations[(a, b)][res_key] for a, b in pairs])
        projected = pca_reduce(vecs, n_components=2).numpy()

        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=sums,
            cmap=cmap,
            s=30,
            edgecolors="k",
            linewidths=0.3,
        )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)

    fig.colorbar(scatter, ax=axes, label="Sum (a+b)", shrink=0.8)
    fig.suptitle("PCA of Residual Stream (colored by sum)", fontsize=13, y=1.02)
    fig.savefig(out_dir / "pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved pca_scatter.png")


def plot_gradient_by_layer(all_grad_norms: dict, n_layers: int, n_pairs: int, out_dir: str) -> None:
    """Stacked bar chart: gradient norm per layer, split by attn vs MLP."""
    layer_grad_attn = defaultdict(float)
    layer_grad_mlp = defaultdict(float)
    for grad_norms in all_grad_norms.values():
        for key, val in grad_norms.items():
            if key.startswith("layer."):
                parts = key.split(".")
                li = int(parts[1])
                comp = parts[2]
                if comp == "self_attn":
                    layer_grad_attn[li] += val
                elif comp == "mlp":
                    layer_grad_mlp[li] += val

    layers = list(range(n_layers))
    attn_vals = [layer_grad_attn[li] / n_pairs for li in layers]
    mlp_vals = [layer_grad_mlp[li] / n_pairs for li in layers]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(layers, attn_vals, label="Attention", color="#3498db")
    ax.bar(layers, mlp_vals, bottom=attn_vals, label="MLP", color="#e74c3c")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean gradient norm")
    ax.set_title("Gradient Magnitude by Layer (stacked: attention + MLP)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "gradient_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved gradient_by_layer.png")


def plot_gradient_by_sum(all_grad_norms: dict, pairs: list, out_dir: str) -> None:
    """Bar chart: mean total gradient norm per sum value."""
    sum_grad: dict[int, list[float]] = defaultdict(list)
    for (a, b), grad_norms in all_grad_norms.items():
        total_norm = sum(grad_norms.values())
        sum_grad[a + b].append(total_norm)

    sorted_sums = sorted(sum_grad)
    means = [np.mean(sum_grad[s]) for s in sorted_sums]
    stds = [np.std(sum_grad[s]) for s in sorted_sums]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sorted_sums, means, yerr=stds, color="#9b59b6", capsize=3, edgecolor="black")
    ax.set_xlabel("Sum (a + b)")
    ax.set_ylabel("Mean total gradient norm")
    ax.set_title("Gradient Norm vs Sum Value")
    ax.set_xticks(sorted_sums)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "gradient_by_sum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved gradient_by_sum.png")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    seed_everything(42)
    device = default_device()
    dtype = default_dtype()
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"

    print(f"Device: {device}  Dtype: {dtype}")

    # --- Load model -----------------------------------------------------------
    print(f"Loading Qwen3-{MODEL_SIZE.upper()} ({dtype_str}) ...")
    model, tokenizer = load_qwen3(MODEL_SIZE, dtype_str=dtype_str, device_map=device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers")

    # --- Build hook targets ---------------------------------------------------
    hook_names: list[str] = []
    for i in range(n_layers):
        hook_names.append(f"model.layers.{i}")  # residual (block output)
        hook_names.append(f"model.layers.{i}.mlp")  # MLP output
        hook_names.append(f"model.layers.{i}.self_attn")  # attention output

    # --- Categorize parameters for gradient analysis --------------------------
    param_groups = categorize_parameters(model, n_layers)

    # --- Storage --------------------------------------------------------------
    all_activations: dict[tuple[int, int], dict[str, Tensor]] = {}
    all_predictions: dict[tuple[int, int], dict] = {}
    all_grad_norms: dict[tuple[int, int], dict[str, float]] = {}

    pairs = [(a, b) for a in A_RANGE for b in B_RANGE]
    n_pairs = len(pairs)
    print(f"\nProcessing {n_pairs} (a, b) pairs ...\n")

    for idx, (a, b) in enumerate(pairs):
        correct = a + b
        prompt = PROMPT_TEMPLATE.format(a=a, b=b)

        # Tokenize (thinking disabled so model predicts answer directly)
        messages, _n_bos, template_kwargs = prepare_messages(prompt, FAMILY, enable_thinking=False)
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, **template_kwargs
        ).to(device)

        # Correct answer token
        answer_tokens = tokenizer.encode(str(correct), add_special_tokens=False)
        if len(answer_tokens) > 1 and idx == 0:
            print(f"  Warning: answer '{correct}' is multi-token: {answer_tokens}")
        answer_token_id = answer_tokens[0]

        # --- Part A: Activations (no grad) ------------------------------------
        recorder = ActivationRecorder()
        with torch.no_grad(), recorder.attach(model, hook_names):
            outputs = model(input_ids)

        # Extract last-token activations
        last_pos = input_ids.shape[1] - 1
        acts: dict[str, Tensor] = {}
        for name, tensor in recorder.activations.items():
            acts[name] = tensor[0, last_pos].cpu()
        all_activations[(a, b)] = acts

        # Prediction info
        logits = outputs.logits[0, last_pos]
        top5 = torch.topk(logits, 5)
        pred_token = top5.indices[0].item()
        pred_str = tokenizer.decode(pred_token)
        correct_in_top5 = answer_token_id in top5.indices.tolist()
        correct_logit_rank = (logits.argsort(descending=True) == answer_token_id).nonzero()
        rank = correct_logit_rank[0].item() if len(correct_logit_rank) > 0 else -1
        all_predictions[(a, b)] = {
            "correct": correct,
            "pred_token": pred_token,
            "pred_str": pred_str,
            "is_correct": pred_token == answer_token_id,
            "correct_in_top5": correct_in_top5,
            "rank": rank,
        }

        # --- Part B: Gradients ------------------------------------------------
        model.zero_grad()
        outputs_grad = model(input_ids)
        logit_correct = outputs_grad.logits[0, last_pos, answer_token_id]
        logit_correct.backward()

        grad_norms: dict[str, float] = {}
        for group_key, params in param_groups.items():
            grad_norms[group_key] = grad_norm_for_group(params)
        all_grad_norms[(a, b)] = grad_norms

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx + 1}/{n_pairs}] {a}+{b}={correct}  pred={pred_str!r}  rank={rank}")

    # ── Analysis ──────────────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("===== Addition Task: Representations & Gradients =====")
    print(
        f"Model: Qwen3-{MODEL_SIZE.upper()} ({n_layers} layers), "
        f"Grid: {len(A_RANGE)}x{len(B_RANGE)} ({n_pairs} pairs)"
    )
    print("=" * 65)

    # --- Prediction Accuracy --------------------------------------------------
    n_correct = sum(1 for p in all_predictions.values() if p["is_correct"])
    n_top5 = sum(1 for p in all_predictions.values() if p["correct_in_top5"])
    print("\n--- Prediction Accuracy ---")
    print(f"  Top-1 accuracy: {n_correct}/{n_pairs} ({100 * n_correct / n_pairs:.1f}%)")
    print(f"  Correct answer in top-5: {n_top5}/{n_pairs}")

    # --- Activation Norms (mean over grid, last token) ------------------------
    print("\n--- Activation Norms (mean over grid, last token) ---")
    print(f"  {'Layer':>5s} | {'Residual':>10s} | {'MLP':>10s} | {'Attention':>10s}")
    print(f"  {'-' * 5}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
    for layer in range(n_layers):
        res_norms = []
        mlp_norms = []
        attn_norms = []
        for acts in all_activations.values():
            res_norms.append(acts[f"model.layers.{layer}"].float().norm().item())
            mlp_norms.append(acts[f"model.layers.{layer}.mlp"].float().norm().item())
            attn_norms.append(acts[f"model.layers.{layer}.self_attn"].float().norm().item())
        res_mean = sum(res_norms) / len(res_norms)
        mlp_mean = sum(mlp_norms) / len(mlp_norms)
        attn_mean = sum(attn_norms) / len(attn_norms)
        print(f"  {layer:>5d} | {res_mean:>10.4f} | {mlp_mean:>10.4f} | {attn_mean:>10.4f}")

    # --- Cosine Similarity Structure ------------------------------------------
    print("\n--- Cosine Similarity Structure (residual stream) ---")
    for layer in ANALYSIS_LAYERS:
        if layer >= n_layers:
            continue
        res_key = f"model.layers.{layer}"
        vecs = torch.stack([all_activations[(a, b)][res_key] for a, b in pairs])
        cos_sim = compute_cosine_sim_matrix(vecs)

        sums = [a + b for a, b in pairs]
        same_sum_sims = []
        diff_sum_sims = []
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                sim_val = cos_sim[i, j].item()
                if sums[i] == sums[j]:
                    same_sum_sims.append(sim_val)
                else:
                    diff_sum_sims.append(sim_val)
        same_mean = sum(same_sum_sims) / len(same_sum_sims) if same_sum_sims else 0
        diff_mean = sum(diff_sum_sims) / len(diff_sum_sims) if diff_sum_sims else 0
        ratio = same_mean / diff_mean if diff_mean != 0 else float("inf")
        print(
            f"  Layer {layer:>2d}: same-sum mean={same_mean:.4f}, "
            f"diff-sum mean={diff_mean:.4f}, ratio={ratio:.2f}"
        )

    # --- PCA of residual stream -----------------------------------------------
    print("\n--- PCA of Residual Stream (top-3 components, variance explained) ---")
    for layer in ANALYSIS_LAYERS:
        if layer >= n_layers:
            continue
        res_key = f"model.layers.{layer}"
        vecs = torch.stack([all_activations[(a, b)][res_key] for a, b in pairs])
        vecs_f = vecs.float()
        vecs_centered = vecs_f - vecs_f.mean(dim=0, keepdim=True)
        total_var = (vecs_centered**2).sum().item()
        if total_var < 1e-10:
            print(f"  Layer {layer:>2d}: near-zero variance")
            continue
        projected = pca_reduce(vecs, n_components=3)
        explained = (projected**2).sum().item() / total_var
        print(f"  Layer {layer:>2d}: top-3 components explain {100 * explained:.1f}% of variance")

    # --- Gradient Norms by Component Type -------------------------------------
    print("\n--- Gradient Norms by Component Type (mean over grid) ---")
    comp_types = {"self_attn": 0.0, "mlp": 0.0, "layernorm": 0.0, "embed_lm_head": 0.0}
    comp_counts = {"self_attn": 0, "mlp": 0, "layernorm": 0, "embed_lm_head": 0}
    for grad_norms in all_grad_norms.values():
        for key, val in grad_norms.items():
            if "self_attn" in key:
                comp_types["self_attn"] += val
                comp_counts["self_attn"] += 1
            elif ".mlp" in key:
                comp_types["mlp"] += val
                comp_counts["mlp"] += 1
            elif "layernorm" in key:
                comp_types["layernorm"] += val
                comp_counts["layernorm"] += 1
            elif key in ("embed_tokens", "lm_head", "final_norm"):
                comp_types["embed_lm_head"] += val
                comp_counts["embed_lm_head"] += 1

    for comp, total in comp_types.items():
        count = comp_counts[comp]
        mean = total / count if count > 0 else 0
        print(f"  {comp:>15s}: {mean:.6f}")

    # --- Gradient Norm by Sum (a+b) -------------------------------------------
    print("\n--- Gradient Norm by Sum (a+b) ---")
    sum_grad: dict[int, list[float]] = defaultdict(list)
    for (a, b), grad_norms in all_grad_norms.items():
        total_norm = sum(grad_norms.values())
        sum_grad[a + b].append(total_norm)
    for s in sorted(sum_grad):
        vals = sum_grad[s]
        mean_val = sum(vals) / len(vals)
        print(f"  sum={s:>2d}: {mean_val:.6f}", end="  ")
        if s % 6 == 0:
            print()
    print()

    # --- Top Layers by Gradient Magnitude -------------------------------------
    print("\n--- Top 5 Layers by Gradient Magnitude ---")
    layer_grad_attn = defaultdict(float)
    layer_grad_mlp = defaultdict(float)
    for grad_norms in all_grad_norms.values():
        for key, val in grad_norms.items():
            if key.startswith("layer."):
                parts = key.split(".")
                li = int(parts[1])
                comp = parts[2]
                if comp == "self_attn":
                    layer_grad_attn[li] += val
                elif comp == "mlp":
                    layer_grad_mlp[li] += val

    layer_total = {
        li: layer_grad_attn.get(li, 0) + layer_grad_mlp.get(li, 0) for li in range(n_layers)
    }
    top_layers = sorted(layer_total, key=lambda x: layer_total[x], reverse=True)[:5]
    for li in top_layers:
        attn_mean = layer_grad_attn[li] / n_pairs
        mlp_mean = layer_grad_mlp[li] / n_pairs
        print(f"  Layer {li:>2d}: attn={attn_mean:.4f}, mlp={mlp_mean:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    out_dir = artifacts_dir() / "addition_representations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating plots ---")
    plot_activation_norms(all_activations, pairs, n_layers, out_dir)
    plot_prediction_grid(all_predictions, out_dir)
    plot_cosine_similarity(all_activations, pairs, n_pairs, ANALYSIS_LAYERS, n_layers, out_dir)
    plot_pca_scatter(all_activations, pairs, ANALYSIS_LAYERS, n_layers, out_dir)
    plot_gradient_by_layer(all_grad_norms, n_layers, n_pairs, out_dir)
    plot_gradient_by_sum(all_grad_norms, pairs, out_dir)

    # ── Save artifacts ────────────────────────────────────────────────────────
    out_path = out_dir / "results.pt"
    torch.save(
        {
            "config": {
                "model_size": MODEL_SIZE,
                "family": FAMILY,
                "a_range": list(A_RANGE),
                "b_range": list(B_RANGE),
                "prompt_template": PROMPT_TEMPLATE,
                "analysis_layers": ANALYSIS_LAYERS,
                "n_layers": n_layers,
            },
            "predictions": all_predictions,
            "activations": all_activations,
            "grad_norms": all_grad_norms,
        },
        out_path,
    )
    print(f"\nSaved artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
