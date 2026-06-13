#!/usr/bin/env python3
"""Addition circuit: find a correctly-solved prompt, then build → prune → identify → ablate.

Biology-paper style analysis for single-digit addition on Qwen3. The 0.6B model
is unreliable at arithmetic under some prompt formats, so this script first
*searches* for a (model, prompt-format, problem) where the model's next-token
prediction is actually the correct sum, preferring the smallest model and the
simplest (zero-shot) format. It then runs the full interpretability protocol on
that case:

1. Build the attribution graph for the correctly-solved ``a+b=`` and prune it.
2. Identify the transcoder features that most influence the answer-digit logit
   (the candidate addition-output / lookup features) and show their label logits.
3. **Validate** by ablating those features on the local replacement model and
   measuring how far the answer logit drops — a causal check, not a correlation.

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
# Try the smallest model first; escalate only if it cannot do the arithmetic.
# fp32 keeps the linearised model faithful; bf16 is used for the larger model to
# fit in memory (the user confirmed bf16 is acceptable on OOM).
MODEL_CANDIDATES: list[tuple[str, str]] = [("0.6b", "fp32"), ("4b", "bf16")]

# Single-digit sums keep the answer a single token (unambiguous target logit).
PROBLEMS: list[tuple[int, int]] = [(2, 3), (4, 5), (1, 6), (3, 4), (6, 2), (5, 4), (7, 2), (4, 4)]

# Prompt formats, simplest (zero-shot) first so we prefer a clean circuit.
FORMATS: list[tuple[str, str]] = [
    ("plain", "{a}+{b}="),
    ("spaces", "{a} + {b} = "),
    ("sum_is", "The sum of {a} and {b} is "),
    ("question", "What is {a}+{b}? Answer with just the number."),
    ("fewshot", "1+1=2\n3+2=5\n7+1=8\n{a}+{b}="),
]

TOP_K_LOGITS = 5
MAX_FEATURE_TARGETS = 500
MIN_EDGE_WEIGHT = 1e-4
NODE_THRESHOLD = 0.7
EDGE_THRESHOLD = 0.9
N_TOP_FEATURES = 8
# ─────────────────────────────────────────────────────────────────────────────


def _chat_input_ids(tokenizer, device, prompt_text: str):
    messages, n_bos, tkw = prepare_messages(prompt_text, "qwen3", enable_thinking=False)
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, **tkw
    ).to(device)
    return input_ids, n_bos


def _next_token_id(model, input_ids) -> int:
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    return int(logits[-1].argmax().item())


def evaluate_formats(
    model, tokenizer, device
) -> dict[str, tuple[float, list[tuple[int, int]], str]]:
    """Return ``{format_name: (accuracy, solved_problems, template)}`` for *model*."""
    out: dict[str, tuple[float, list[tuple[int, int]], str]] = {}
    for name, tmpl in FORMATS:
        solved: list[tuple[int, int]] = []
        for a, b in PROBLEMS:
            input_ids, _ = _chat_input_ids(tokenizer, device, tmpl.format(a=a, b=b))
            dec = tokenizer.decode(_next_token_id(model, input_ids))
            if dec.strip() == str(a + b):
                solved.append((a, b))
        out[name] = (len(solved) / len(PROBLEMS), solved, tmpl)
    return out


def pick_best(evald) -> tuple[str, str, tuple[int, int] | None, float]:
    """Highest-accuracy format (ties broken toward the simpler/earlier format)."""
    best_acc, best_name, best_tmpl, best_solved = -1.0, "", "", []
    for name, tmpl in FORMATS:
        acc, solved, _ = evald[name]
        if acc > best_acc:
            best_acc, best_name, best_tmpl, best_solved = acc, name, tmpl, solved
    problem = best_solved[0] if best_solved else None
    return best_name, best_tmpl, problem, best_acc


def build_circuit(model, tokenizer, tc, repo_id, size, tmpl, a, b, out_dir) -> None:
    """Build → prune → identify → ablate for the prompt ``tmpl.format(a, b)``."""
    device = next(model.parameters()).device
    prompt = tmpl.format(a=a, b=b)
    input_ids, n_bos = _chat_input_ids(tokenizer, device, prompt)
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    expected = str(a + b)
    answer_id = _next_token_id(model, input_ids)
    answer_str = tokenizer.decode(answer_id)
    print(f"\nPrompt {prompt!r}  ->  predicts {answer_str!r}  (expected {expected!r})")
    print(f"Tokens ({len(tokens)}): {tokens}")

    # --- 1. Build + prune -----------------------------------------------------
    print("\nBuilding attribution graph ...")
    graph = build_attribution_graph(
        model,
        tc,
        input_ids,
        n_bos_tokens=n_bos,
        top_k_logits=TOP_K_LOGITS,
        max_feature_targets=MAX_FEATURE_TARGETS,
        min_edge_weight=MIN_EDGE_WEIGHT,
    )
    print(f"  raw graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    pruned = prune_graph(graph, node_threshold=NODE_THRESHOLD, edge_threshold=EDGE_THRESHOLD)
    pg = pruned.graph
    print(f"  pruned graph: {len(pg.nodes)} nodes, {len(pg.edges)} edges")

    # --- 2. Identify features driving the answer logit ------------------------
    logit_idxs = [i for i, n in enumerate(pg.nodes) if n.node_type == "logit"]
    answer_logit_idx = next(
        (i for i in logit_idxs if pg.nodes[i].token_id == answer_id),
        max(logit_idxs, key=lambda i: pg.nodes[i].activation) if logit_idxs else None,
    )
    if answer_logit_idx is None:
        print("No logit node survived pruning; aborting feature identification.")
        return

    direct = [
        (e.source, e.weight)
        for e in pg.edges
        if e.target == answer_logit_idx and pg.nodes[e.source].node_type == "feature"
    ]
    direct.sort(key=lambda t: abs(t[1]), reverse=True)
    top = direct[:N_TOP_FEATURES]

    by_layer: dict[int, list[int]] = {}
    for src_idx, _ in top:
        nd = pg.nodes[src_idx]
        by_layer.setdefault(nd.layer, []).append(nd.feature_idx)
    labels = {layer: load_feature_labels(repo_id, layer, idxs) for layer, idxs in by_layer.items()}

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
    if candidate_ablations:
        single = run_feature_ablation(
            model, tc, input_ids, [candidate_ablations[0]], n_bos_tokens=n_bos
        )
        eff = ablation_logit_effect(single, [answer_id])[answer_id]
        f0 = candidate_ablations[0]
        print(
            f"  ablate L{f0.layer} f{f0.feature_idx} (pos {f0.position}): "
            f"Δlogit({answer_str!r}) = {eff:+.4f}"
        )

    joint = run_feature_ablation(model, tc, input_ids, candidate_ablations, n_bos_tokens=n_bos)
    joint_eff = ablation_logit_effect(joint, [answer_id])[answer_id]
    new_top = int(joint.ablated_logits[-1].argmax().item())
    print(
        f"  ablate all {len(candidate_ablations)} features: "
        f"Δlogit({answer_str!r}) = {joint_eff:+.4f}  | new top token = {tokenizer.decode(new_top)!r}"
    )

    # --- Save artifacts -------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    logit_token_strs = {
        str(n.token_id): tokenizer.decode(n.token_id) for n in pg.nodes if n.node_type == "logit"
    }
    graph_dict = graph_to_dict(
        pg,
        influence_scores=pruned.influence_scores,
        prompt=prompt,
        model=f"qwen3-{size}",
        tokens=tokens,
        n_bos_tokens=n_bos,
        logit_token_strs=logit_token_strs,
        answer_token=answer_str,
        expected_answer=expected,
        correct=(answer_str.strip() == expected),
    )
    stem = f"addition_graph_qwen3-{size}"
    (out_dir / f"{stem}.json").write_text(json.dumps(graph_dict, indent=2))
    render_graph_html(
        graph_dict,
        out_dir / f"{stem}.html",
        title=f"Addition {prompt!r} -> {answer_str!r} (qwen3-{size})",
    )
    print(f"\nSaved graph + visualization to {out_dir} ({stem}.json / .html)")


def main() -> None:
    device = default_device()
    out_dir = artifacts_dir() / "addition_circuit"

    chosen = None
    for size, dtype_str in MODEL_CANDIDATES:
        dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16
        print(f"\n{'=' * 70}\nTrying Qwen3-{size} ({dtype_str})\n{'=' * 70}")
        model, tokenizer = load_qwen3(size, dtype_str=dtype_str, device_map=device)
        model.eval()

        evald = evaluate_formats(model, tokenizer, device)
        for name, _ in FORMATS:
            acc, solved, _ = evald[name]
            print(f"  format {name:>9}: accuracy {acc * 100:5.1f}%  solved={solved}")

        name, tmpl, problem, acc = pick_best(evald)
        if problem is not None:
            print(f"\n  -> Qwen3-{size}, format {name!r} (acc {acc * 100:.0f}%), problem {problem}")
            loaded = load_transcoder(f"qwen3-{size}", device=device, dtype=dtype)
            chosen = (size, model, tokenizer, loaded.transcoder, loaded.repo_id, tmpl, problem)
            break

        print(f"  Qwen3-{size} solved nothing in any format; trying next model.")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if chosen is None:
        print("\nNo candidate model produced a correct addition; aborting.")
        return

    size, model, tokenizer, tc, repo_id, tmpl, (a, b) = chosen
    build_circuit(model, tokenizer, tc, repo_id, size, tmpl, a, b, out_dir)


if __name__ == "__main__":
    main()
