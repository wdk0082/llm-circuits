#!/usr/bin/env python3
"""Addition circuits: find a model that adds, then graph SEVERAL examples.

Biology-paper style analysis for single-digit addition on Qwen3. The 0.6B model
is unreliable at arithmetic, so this script first *searches* for a (model,
prompt-format) where the model's next-token prediction is the correct sum,
preferring the smallest model and the simplest format. It then runs the full
interpretability protocol on several correctly-solved problems:

  build attribution graph -> prune -> attach feature labels -> identify the
  features driving the answer-digit logit -> **negative-steer** them to causally confirm.

Output: one self-contained ``addition_suite_qwen3-<size>.html`` with a dropdown
to switch between examples (each an interactive graph), plus per-example
JSON/HTML. Run:

    uv run python examples/addition_circuit.py
"""

from __future__ import annotations

import json

import torch

from llm_circuits.circuits.attribution_graph import build_attribution_graph
from llm_circuits.circuits.graph_pruning import graph_to_dict, prune_graph
from llm_circuits.circuits.interventions import (
    ablation_logit_effect,
    ablation_prob_effect,
    negative_steer,
    run_feature_intervention,
)
from llm_circuits.circuits.visualization import render_graph_html_str, render_suite_html
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder
from llm_circuits.transcoders.feature_labels import load_feature_labels

# ── Config ───────────────────────────────────────────────────────────────────
# Qwen3-0.6B cannot reliably add (verified: 0% across formats), so go straight to 4B.
# We still search prompt formats to pick the cleanest one the model actually solves.
MODEL_CANDIDATES: list[tuple[str, str]] = [("4b", "bf16")]

# Single-digit sums (answer is a single token); chosen to span distinct sums.
PROBLEMS: list[tuple[int, int]] = [(1, 2), (2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 5)]

# Prompt formats, simplest (zero-shot) first so we prefer a clean circuit.
FORMATS: list[tuple[str, str]] = [
    ("plain", "{a}+{b}="),
    ("spaces", "{a} + {b} = "),
    ("sum_is", "The sum of {a} and {b} is "),
    ("question", "What is {a}+{b}? Answer with just the number."),
    ("fewshot", "1+1=2\n3+2=5\n7+1=8\n{a}+{b}="),
]

N_EXAMPLES = 6  # how many solved problems to graph
TOP_K_LOGITS = 5
# Cap active-feature NODES (bounds the prune adjacency matrix), then compute the FULL
# edge matrix over them (targets=None) -- affordable now that the edge backward is batched.
MAX_FEATURE_NODES = 8000
MAX_FEATURE_TARGETS = None
MIN_EDGE_WEIGHT = 1e-4
NODE_THRESHOLD = 0.7
EDGE_THRESHOLD = 0.9
N_TOP_FEATURES = 8  # features steered per example
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


def evaluate_formats(model, tokenizer, device):
    """Return ``{format_name: (accuracy, solved_problems, template)}`` for *model*."""
    out: dict[str, tuple[float, list[tuple[int, int]], str]] = {}
    for name, tmpl in FORMATS:
        solved: list[tuple[int, int]] = []
        for a, b in PROBLEMS:
            input_ids, _ = _chat_input_ids(tokenizer, device, tmpl.format(a=a, b=b))
            if tokenizer.decode(_next_token_id(model, input_ids)).strip() == str(a + b):
                solved.append((a, b))
        out[name] = (len(solved) / len(PROBLEMS), solved, tmpl)
    return out


def pick_best(evald):
    """Highest-accuracy format (ties broken toward the simpler/earlier format)."""
    best = (-1.0, "", "", [])
    for name, tmpl in FORMATS:
        acc, solved, _ = evald[name]
        if acc > best[0]:
            best = (acc, name, tmpl, solved)
    acc, name, tmpl, solved = best
    return name, tmpl, solved, acc


def build_circuit(model, tokenizer, tc, repo_id, size, tmpl, a, b, out_dir) -> dict | None:
    """Build → prune → label → identify → negative-steer for one problem; save + return entry."""
    device = next(model.parameters()).device
    prompt = tmpl.format(a=a, b=b)
    input_ids, n_bos = _chat_input_ids(tokenizer, device, prompt)
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    expected = str(a + b)
    answer_id = _next_token_id(model, input_ids)
    answer_str = tokenizer.decode(answer_id)
    correct = answer_str.strip() == expected
    print(f"\n[{a}+{b}={expected}] predicts {answer_str!r} ({'correct' if correct else 'WRONG'})")

    graph = build_attribution_graph(
        model,
        tc,
        input_ids,
        n_bos_tokens=n_bos,
        top_k_logits=TOP_K_LOGITS,
        max_feature_targets=MAX_FEATURE_TARGETS,
        max_feature_nodes=MAX_FEATURE_NODES,
        min_edge_weight=MIN_EDGE_WEIGHT,
    )
    pruned = prune_graph(graph, node_threshold=NODE_THRESHOLD, edge_threshold=EDGE_THRESHOLD)
    pg = pruned.graph
    print(
        f"  graph: {len(graph.nodes)}→{len(pg.nodes)} nodes, {len(graph.edges)}→{len(pg.edges)} edges"
    )

    # Attach feature labels to every feature node (so the viz shows meaning).
    feat_by_layer: dict[int, list[int]] = {}
    for nd in pg.nodes:
        if nd.node_type == "feature":
            feat_by_layer.setdefault(nd.layer, []).append(nd.feature_idx)
    label_lookup: dict[tuple[int, int], dict] = {}
    for layer, idxs in feat_by_layer.items():
        for fidx, lab in load_feature_labels(repo_id, layer, idxs).items():
            label_lookup[(layer, fidx)] = lab.to_dict()
    for nd in pg.nodes:
        if nd.node_type == "feature":
            nd.label = label_lookup.get((nd.layer, nd.feature_idx))

    # Identify features feeding the answer logit.
    logit_idxs = [i for i, n in enumerate(pg.nodes) if n.node_type == "logit"]
    answer_logit_idx = next(
        (i for i in logit_idxs if pg.nodes[i].token_id == answer_id),
        max(logit_idxs, key=lambda i: pg.nodes[i].activation) if logit_idxs else None,
    )
    if answer_logit_idx is None:
        print("  no logit node survived pruning; skipping example")
        return None
    direct = [
        (e.source, e.weight)
        for e in pg.edges
        if e.target == answer_logit_idx and pg.nodes[e.source].node_type == "feature"
    ]
    direct.sort(key=lambda t: abs(t[1]), reverse=True)
    top = direct[:N_TOP_FEATURES]
    steers = [
        negative_steer(pg.nodes[i].layer, pg.nodes[i].feature_idx, position=pg.nodes[i].position)
        for i, _ in top
    ]

    def _top1(nd):
        return nd.label["top_logits"][0] if (nd.label and nd.label["top_logits"]) else "?"

    top3 = "; ".join(
        f"L{pg.nodes[i].layer} f{pg.nodes[i].feature_idx} ({_top1(pg.nodes[i])!s})"
        for i, _ in top[:3]
    )
    print(f"  top features: {top3}")

    # Validate by negative steering (the paper's protocol): steer each feature to
    # -1x its clean value via constrained patching, and report the logit delta +
    # post-softmax probability transition for the answer token.
    eff = 0.0
    base_p = top_p = 0.0
    if steers:
        single = run_feature_intervention(model, tc, input_ids, [steers[0]], n_bos_tokens=n_bos)
        eff = ablation_logit_effect(single, [answer_id])[answer_id]
        base_p, top_p = ablation_prob_effect(single, [answer_id])[answer_id]
    joint = run_feature_intervention(model, tc, input_ids, steers, n_bos_tokens=n_bos)
    joint_eff = ablation_logit_effect(joint, [answer_id])[answer_id]
    jb_p, all_p = ablation_prob_effect(joint, [answer_id])[answer_id]
    if not steers:
        base_p = jb_p
    new_top = tokenizer.decode(int(joint.ablated_logits[-1].argmax().item()))
    print(f"  steer top:  Δlogit={eff:+.2f}  p({answer_str!r}) {base_p:.3f}→{top_p:.3f}")
    print(
        f"  steer all {len(steers)}: Δlogit={joint_eff:+.2f}  "
        f"p {base_p:.3f}→{all_p:.3f}  new top {new_top!r}"
    )

    # Serialise + render this example.
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
        correct=correct,
        steering={
            "method": "negative_steer_constrained",
            "n_steered": len(steers),
            "top_logit_delta": eff,
            "all_logit_delta": joint_eff,
            "answer_prob_baseline": base_p,
            "answer_prob_steer_top": top_p,
            "answer_prob_steer_all": all_p,
            "new_top_after_steering": new_top,
        },
    )
    stem = f"addition_graph_qwen3-{size}_{a}plus{b}"
    (out_dir / f"{stem}.json").write_text(json.dumps(graph_dict, indent=2))
    # The per-example graph is embedded in the combined suite HTML below (and the
    # JSON above is what downstream scripts read), so we don't write a standalone
    # per-example .html — it would just duplicate the suite and clutter the dir.
    graph_html = render_graph_html_str(graph_dict, title=f"{prompt} → {answer_str!r}")

    mark = "✓" if correct else "✗"
    summary = (
        f"<b>{a}+{b}={expected}</b> → '{answer_str}' {mark} &nbsp;|&nbsp; "
        f"p('{answer_str}'): {base_p:.2f} -> {top_p:.2f} (-top) -> {all_p:.2f} (-all {len(steers)}) "
        f"&nbsp;|&nbsp; Δlogit {eff:+.1f} / {joint_eff:+.1f} &nbsp;|&nbsp; "
        f"top: {top3} &nbsp;|&nbsp; new top '{new_top}'"
    )
    return {
        "label": f"{a}+{b}={expected} ({mark})",
        "summary": summary,
        "graph_html": graph_html,
        "row": (
            f"{a}+{b}",
            expected,
            answer_str,
            mark,
            eff,
            joint_eff,
            base_p,
            top_p,
            all_p,
            new_top,
        ),
    }


def build_suite(model, tokenizer, tc, repo_id, size, tmpl, problems, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, rows = [], []
    for a, b in problems:
        res = build_circuit(model, tokenizer, tc, repo_id, size, tmpl, a, b, out_dir)
        if res is not None:
            entries.append({k: res[k] for k in ("label", "summary", "graph_html")})
            rows.append(res["row"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not entries:
        print("No examples built; nothing to assemble.")
        return

    suite_path = out_dir / f"addition_suite_qwen3-{size}.html"
    render_suite_html(entries, suite_path, title=f"Addition circuits — Qwen3-{size}")

    print("\n" + "=" * 78)
    print(f"Summary ({len(rows)} examples, Qwen3-{size})")
    print("=" * 78)
    print(
        f"  {'prob':>6} {'pred':>5} {'ok':>3} {'Δtop':>7} {'Δall':>7} "
        f"{'p:base→top→all':>22} {'newtop':>7}"
    )
    for prob, _exp, pred, mark, eff, joint_eff, base_p, top_p, all_p, new_top in rows:
        ptrans = f"{base_p:.2f}→{top_p:.2f}→{all_p:.2f}"
        print(
            f"  {prob:>6} {pred:>5} {mark:>3} {eff:>7.2f} {joint_eff:>7.2f} {ptrans:>22} {new_top!r:>7}"
        )
    print(f"\nSaved suite viewer: {suite_path}")


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

        name, tmpl, solved, acc = pick_best(evald)
        if solved:
            print(
                f"\n  -> Qwen3-{size}, format {name!r} (acc {acc * 100:.0f}%), {len(solved)} solved"
            )
            loaded = load_transcoder(f"qwen3-{size}", device=device, dtype=dtype)
            chosen = (size, model, tokenizer, loaded.transcoder, loaded.repo_id, tmpl, solved)
            break

        print(f"  Qwen3-{size} solved nothing in any format; trying next model.")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if chosen is None:
        print("\nNo candidate model produced a correct addition; aborting.")
        return

    size, model, tokenizer, tc, repo_id, tmpl, solved = chosen
    build_suite(model, tokenizer, tc, repo_id, size, tmpl, solved[:N_EXAMPLES], out_dir)


if __name__ == "__main__":
    main()
