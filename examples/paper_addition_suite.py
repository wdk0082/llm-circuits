"""PAPER-EXACT addition reproduction suite (biology.html §Addition + methods supplement).

Stages (all use the paper's raw ``calc: a+b=`` format unless its accuracy collapses,
in which case the chat fallback is used and reported):

  accuracy      — the paper's 10,000-prompt grid (a,b in [0,99]).
  graphs        — attribution graphs for the studied prompt (magnitude digit + ones
                  digit). Default calc: 36+59= (the paper's); if the model answers it
                  incorrectly, the nearest CORRECT pair with the same digit classes
                  (a%10, b%10, a+b preserved) is studied instead — the paper's premise
                  is a correctly-completed prompt.
  grids         — operand-grid heatmaps + taxonomy for answer features, mid-layer
                  features (add-function / lookup hunt) and operand-token input
                  features (the _6 / _9 / ~magnitude supernodes). (Fig A1/A2)
  interventions — methods-supplement set: suppress _6, suppress _9 (expect ones-digit
                  shifts a la 98/91), inhibit magnitude inputs with readout of the
                  downstream lookup/sum features (% of baseline).
  steer_compare — negative-steer lookup vs sum features (paper -2x => m=-3); compare
                  output smear widths (paper: lookup ~5 wide, sum wider).
  polymer       — the paper's journal-citation prompt: completion, graph, feature
                  overlap with the calc circuit (the reuse claim), -2x suppression
                  of the shared features. (Fig A4/A5)
  intermediate  — assert (4 + 5) * 3 == : completion, graph, computed-9 hunt. (Fig A6)
  introspection — the two-turn dialogue (model narrates its method).
  corpus        — dataset examples of the top lookup/sum features (Fig A3; streams
                  C4 via `datasets`).

Artifacts land in ``$LLM_CIRCUITS_ARTIFACTS_DIR/paper_addition/<size>/``.

Usage:
    uv run python examples/paper_addition_suite.py --size 4b --stages all
    gcp/launch.sh examples/paper_addition_suite.py --size 4b --stages accuracy,graphs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import addition_paper as P
import helper as H

from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

ALL_STAGES = [
    "accuracy",
    "graphs",
    "grids",
    "interventions",
    "steer_compare",
    "polymer",
    "intermediate",
    "introspection",
    "corpus",
]
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[suite +{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


def save_json(out_dir: Path, name: str, obj) -> None:
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    log(f"wrote {path}")


def pick_studied_pair(a0: int, b0: int, correct: np.ndarray) -> tuple[int, int, bool]:
    """The paper studies a prompt the model answers CORRECTLY (Haiku: 36+59 -> 95).

    If the model answers (a0, b0) correctly, keep it.  Otherwise pick the nearest
    two-digit pair with the SAME digit classes — a % 10, b % 10 and a + b all equal to
    the requested pair's (preserving the paper's ``_6 + _9 -> sum = _95`` lookup/sum
    structure) — that the model does answer correctly.
    """
    if correct[a0, b0]:
        return a0, b0, False
    cands = [
        (a, b)
        for a in range(10, 100)
        for b in range(10, 100)
        if a % 10 == a0 % 10 and b % 10 == b0 % 10 and a + b == a0 + b0 and correct[a, b]
    ]
    if not cands:
        raise SystemExit(
            f"model answers {a0}+{b0} incorrectly and no correct same-class pair exists"
        )
    a, b = min(cands, key=lambda ab: abs(ab[0] - a0) + abs(ab[1] - b0))
    return a, b, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="4b")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--a", type=int, default=36, help="left operand of the studied prompt")
    ap.add_argument("--b", type=int, default=59, help="right operand of the studied prompt")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--graph-feature-nodes", type=int, default=8000)
    ap.add_argument("--grid-batch", type=int, default=256)
    ap.add_argument("--corpus-docs", type=int, default=2000)
    args = ap.parse_args()
    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")

    device = default_device()
    out_dir = artifacts_dir() / "paper_addition" / args.size
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"device={device} size={args.size} stages={stages} -> {out_dir}")

    model, tokenizer = load_qwen3(args.size, dtype_str=args.dtype, device_map=device)
    model.eval()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tc = load_transcoder(f"qwen3-{args.size}", device=device, dtype=dtype).transcoder
    log(f"loaded Qwen3-{args.size} + transcoders ({len(tc)} layers)")

    A_VALS = list(range(100))
    B_VALS = list(range(100))
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    # ------------------------------------------------------------------
    # accuracy — paper format first, chat fallback
    # ------------------------------------------------------------------
    if "accuracy" in stages:
        acc = {}
        for style in ("calc", "chat"):
            correct, _pred = H.accuracy_grid(
                model, tokenizer, A_VALS, B_VALS, batch_size=args.grid_batch, style=style
            )
            acc[style] = float(correct.mean())
            log(f"accuracy[{style}] = {acc[style]:.1%}")
            np.save(out_dir / f"accuracy_{style}.npy", correct)
            if style == "calc" and acc[style] >= 0.5:
                break  # paper format works; skip the fallback measurement
        style = "calc" if acc.get("calc", 0.0) >= 0.5 else "chat"
        state["style"] = style
        state["accuracy"] = acc
        save_json(out_dir, "state", state)
        log(f"using style={style!r} for all subsequent stages")
    style = state.get("style", "calc")

    # ------------------------------------------------------------------
    # studied pair — must be a prompt the model answers correctly (paper premise)
    # ------------------------------------------------------------------
    if "pair" in state and [args.a, args.b] == state.get("pair_requested", [args.a, args.b]):
        A0, B0 = state["pair"]  # resume: reuse the pair earlier stages selected
    else:
        A0, B0 = args.a, args.b
        acc_path = out_dir / f"accuracy_{style}.npy"
        if acc_path.exists():
            A0, B0, switched = pick_studied_pair(args.a, args.b, np.load(acc_path))
            if switched:
                log(
                    f"NOTE: model answers {args.a}+{args.b} INCORRECTLY in style={style!r}; "
                    f"studying the nearest correct same-class pair {A0}+{B0}={A0 + B0} instead"
                )
        else:
            log(f"WARNING: no {acc_path.name}; using pair {A0}+{B0} unverified")
        state["pair_requested"] = [args.a, args.b]
        state["pair"] = [A0, B0]
        save_json(out_dir, "state", state)
    log(f"studied pair: {A0}+{B0}={A0 + B0}")

    # ------------------------------------------------------------------
    # graphs — calc: 36+59= (both digit circuits)
    # ------------------------------------------------------------------
    graphs: dict = {}
    ids: dict = {}
    answers: dict = {}
    need_graphs = {"graphs", "grids", "interventions", "steer_compare", "polymer"} & set(stages)
    if need_graphs:
        for target in ("first", "ones"):
            t = time.perf_counter()
            d, aid, iid = H.build_addition_graph(
                model,
                tc,
                tokenizer,
                A0,
                B0,
                target=target,
                style=style,
                max_feature_nodes=args.graph_feature_nodes,
                out_html=str(out_dir / f"graph_{A0}p{B0}_{target}.html"),
            )
            graphs[target], answers[target], ids[target] = d, aid, iid
            log(
                f"graph[{target}]: answer={tokenizer.decode([aid])!r} "
                f"({time.perf_counter() - t:.0f}s)"
            )

    # ------------------------------------------------------------------
    # grids — taxonomy for answer + mid-layer + input features
    # ------------------------------------------------------------------
    if "grids" in stages:
        n_layers = len(tc)
        probe_first = "last" if style == "calc" else "first"

        groups: dict[str, list[tuple[int, int]]] = {}
        ans_first = H.answer_features(graphs["first"], top_k=8, numeric_only=True)
        ans_ones = H.answer_features(graphs["ones"], top_k=8, numeric_only=True)
        groups["answer_first"] = [(f["layer"], f["feature_idx"]) for f in ans_first]
        groups["answer_ones"] = [(f["layer"], f["feature_idx"]) for f in ans_ones]

        # mid-layer hunt: influential features in the middle band of the ONES graph
        mid = [
            n
            for n in graphs["ones"]["nodes"]
            if n["node_type"] == "feature" and n_layers // 4 <= n["layer"] < 3 * n_layers // 4
        ]
        mid.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
        groups["midlayer"] = [(n["layer"], n["feature_idx"]) for n in mid[:16]]

        # input supernodes: EARLY-layer features on the operand digit tokens (ONES graph
        # positions).  The layer cap matters: the paper's _6/_9/~magnitude input features
        # are detokenization-level, and without it late-layer aggregation features at the
        # operand positions win the influence ranking (leaving _9 empty).
        pos = P.digit_token_positions(tokenizer, ids["ones"], A0, B0)
        state["digit_positions"] = pos
        lmax_input = max(1, n_layers // 4)
        groups["input_a"] = [
            (L, i)
            for (L, i, _) in P.position_features(
                graphs["ones"], pos["a_digits"], 16, max_layer=lmax_input
            )
        ]
        groups["input_b"] = [
            (L, i)
            for (L, i, _) in P.position_features(
                graphs["ones"], pos["b_digits"], 16, max_layer=lmax_input
            )
        ]

        all_feats = sorted({f for g in groups.values() for f in g})
        log(f"grids: probing {len(all_feats)} features over {len(A_VALS) * len(B_VALS)} prompts")
        grids_first = H.feature_grids(
            model,
            tc,
            all_feats,
            A_VALS,
            B_VALS,
            tokenizer,
            batch_size=args.grid_batch,
            probe=probe_first,
            style=style,
        )
        grids_ones = H.feature_grids(
            model,
            tc,
            all_feats,
            A_VALS,
            B_VALS,
            tokenizer,
            batch_size=args.grid_batch,
            probe="ones",
            style=style,
        )
        # Input features live ON the operand digit tokens, not the final position: probe
        # them at their peak over prompt positions (with style="calc", grids_first reads
        # the "=" token, where input-token features are silent).
        grids_peak = (
            grids_first
            if probe_first == "first"
            else H.feature_grids(
                model,
                tc,
                all_feats,
                A_VALS,
                B_VALS,
                tokenizer,
                batch_size=args.grid_batch,
                probe="first",
                style=style,
            )
        )

        def grids_for(gname: str):
            if gname in ("answer_ones", "midlayer"):
                return grids_ones
            if gname in ("input_a", "input_b"):
                return grids_peak
            return grids_first

        np.savez_compressed(
            out_dir / "grids.npz",
            **{f"first_L{L}_f{i}": grids_first[(L, i)] for (L, i) in all_feats},
            **{f"ones_L{L}_f{i}": grids_ones[(L, i)] for (L, i) in all_feats},
            **{f"peak_L{L}_f{i}": grids_peak[(L, i)] for (L, i) in all_feats},
        )

        taxonomy: dict[str, dict] = {}
        for gname, feats in groups.items():
            grids = grids_for(gname)
            taxonomy[gname] = {
                f"L{L}f{i}": H.periodicity_report(grids[(L, i)], A_VALS, B_VALS) for (L, i) in feats
            }
            counts: dict[str, int] = {}
            for rep in taxonomy[gname].values():
                key = str(rep["label"]).split("(")[0]
                counts[key] = counts.get(key, 0) + 1
            log(f"taxonomy[{gname}]: {counts}")
        save_json(out_dir, "taxonomy", taxonomy)

        # panel figures per group (paper Fig A1-style)
        for gname, feats in groups.items():
            if not feats:
                continue
            grids = grids_for(gname)
            cols = 4
            rows = (len(feats) + cols - 1) // cols
            fig, axes = H.plt.subplots(rows, cols, figsize=(3.6 * cols, 3.1 * rows))
            for ax, (L, i) in zip(np.ravel(axes), feats, strict=False):
                rep = H.periodicity_report(grids[(L, i)], A_VALS, B_VALS)
                H.plot_grid(
                    grids[(L, i)],
                    A_VALS,
                    B_VALS,
                    ax=ax,
                    mark=(A0, B0),
                    title=f"L{L} f{i}\n{rep['label']}",
                )
            for ax in np.ravel(axes)[len(feats) :]:
                ax.axis("off")
            fig.suptitle(f"{gname} ({style})", y=1.005)
            fig.tight_layout()
            fig.savefig(out_dir / f"grids_{gname}.png", dpi=130, bbox_inches="tight")
            H.plt.close("all")

        # persist selections for later stages
        def _sel(gname, pred):
            return [
                [int(x) for x in key[1:].split("f")]
                for key, rep in taxonomy[gname].items()
                if pred(rep)
            ]

        state["lookup_feats"] = _sel(
            "midlayer", lambda r: str(r["label"]).startswith("lookup")
        ) + _sel("answer_ones", lambda r: str(r["label"]).startswith("lookup"))
        state["sum_feats"] = _sel("answer_ones", lambda r: str(r["label"]).startswith("mod10-sum"))
        state["input_6"] = _sel(
            "input_a",
            lambda r: (
                str(r["label"]).startswith(("mod10-a", "lookup")) and r.get("a_top") == A0 % 10
            ),
        )
        state["input_9"] = _sel(
            "input_b",
            lambda r: (
                str(r["label"]).startswith(("mod10-b", "lookup")) and r.get("b_top") == B0 % 10
            ),
        )
        state["input_mag"] = _sel(
            "input_a", lambda r: str(r["label"]) in ("magnitude-diag", "mixed")
        ) + _sel("input_b", lambda r: str(r["label"]) in ("magnitude-diag", "mixed"))
        save_json(out_dir, "state", state)
        log(
            f"selected: lookup={len(state['lookup_feats'])} sum={len(state['sum_feats'])} "
            f"_6={len(state['input_6'])} _9={len(state['input_9'])} mag={len(state['input_mag'])}"
        )

    # helpers for intervention stages -----------------------------------
    def feats_with_acts(keys: list[list[int]], graph_key: str = "ones"):
        by_key = {
            (n["layer"], n["feature_idx"]): float(n.get("activation", 0.0))
            for n in graphs[graph_key]["nodes"]
            if n["node_type"] == "feature"
        }
        return [(L, i, by_key.get((L, i), 0.0)) for L, i in keys]

    if {"interventions", "steer_compare"} & set(stages) and "lookup_feats" not in state:
        raise SystemExit("run the 'grids' stage first (feature selections missing)")

    # ------------------------------------------------------------------
    # interventions — methods-supplement suppressions with feature readout
    # ------------------------------------------------------------------
    if "interventions" in stages:
        last = ids["ones"].shape[1] - 1
        readout = [(L, i, last) for L, i in state["lookup_feats"] + state["sum_feats"]]
        results = {}
        for name, keys, pos_list in [
            ("suppress_6", state["input_6"], state["digit_positions"]["a_digits"][-1:]),
            ("suppress_9", state["input_9"], state["digit_positions"]["b_digits"][-1:]),
            ("inhibit_magnitude", state["input_mag"], None),
        ]:
            if not keys:
                results[name] = {"skipped": "no features selected"}
                log(f"{name}: SKIPPED (no features)")
                continue
            pos = pos_list[0] if pos_list else None
            for m in (-1.0, -2.0):  # ablate and sign-flip (paper 'suppression')
                rep = P.steer_and_report(
                    model,
                    tc,
                    ids["ones"],
                    feats_with_acts(keys),
                    tokenizer,
                    m=m,
                    position=pos,
                    readout=readout,
                )
                results[f"{name}_m{m:g}"] = rep
                log(f"{name} m={m:g}: steered_top={rep['steered_top'][:3]}")
        save_json(out_dir, "interventions", results)

    # ------------------------------------------------------------------
    # steer_compare — lookup vs sum negative steering (paper -2x => m=-3)
    # ------------------------------------------------------------------
    if "steer_compare" in stages:
        last = ids["ones"].shape[1] - 1
        results = {}
        for name, keys in [("lookup", state["lookup_feats"]), ("sum", state["sum_feats"])]:
            if not keys:
                results[name] = {"skipped": "no features selected"}
                continue
            rep = P.steer_and_report(
                model, tc, ids["ones"], feats_with_acts(keys), tokenizer, m=-3.0, position=last
            )
            results[name] = rep
            log(
                f"steer_compare[{name}] m=-3: width {rep['baseline_digits']['width']:.2f} -> "
                f"{rep['steered_digits']['width']:.2f}"
            )
        save_json(out_dir, "steer_compare", results)

    # ------------------------------------------------------------------
    # polymer — citation reuse (Fig A4/A5)
    # ------------------------------------------------------------------
    if "polymer" in stages:
        device_ids = H.tokenize_raw(tokenizer, P.POLYMER_PROMPT, next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(
                device_ids,
                attention_mask=torch.ones_like(device_ids),
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0, device_ids.shape[1] :], skip_special_tokens=True)
        log(f"polymer completion: {completion!r} (paper: '995')")
        pg, aid, pids = P.build_graph_raw(
            model,
            tc,
            tokenizer,
            P.POLYMER_PROMPT,
            size_key=args.size,
            max_feature_nodes=args.graph_feature_nodes,
            out_html=str(out_dir / "graph_polymer.html"),
            title="polymer citation",
        )
        calc_feats = {
            (n["layer"], n["feature_idx"])
            for n in graphs["ones"]["nodes"]
            if n["node_type"] == "feature"
        }
        poly_feats = {
            (n["layer"], n["feature_idx"]) for n in pg["nodes"] if n["node_type"] == "feature"
        }
        shared = calc_feats & poly_feats
        shared_selected = [
            list(k)
            for k in shared
            if list(k) in state.get("lookup_feats", []) + state.get("sum_feats", [])
        ]
        poly_acts = {
            (n["layer"], n["feature_idx"]): float(n.get("activation", 0.0))
            for n in pg["nodes"]
            if n["node_type"] == "feature"
        }
        result = {
            "completion": completion,
            "answer_token": tokenizer.decode([aid]),
            "n_calc_ones_features": len(calc_feats),
            "n_polymer_features": len(poly_feats),
            "n_shared": len(shared),
            "shared_lookup_or_sum": shared_selected,
        }
        if shared_selected:
            feats = [(L, i, poly_acts.get((L, i), 0.0)) for L, i in shared_selected]
            rep = P.steer_and_report(
                model, tc, pids, feats, tokenizer, m=-3.0, position=pids.shape[1] - 1
            )
            result["suppress_shared_m-3"] = rep
            log(f"polymer suppress shared: {rep['steered_top'][:3]}")
        save_json(out_dir, "polymer", result)

    # ------------------------------------------------------------------
    # intermediate — assert (4 + 5) * 3 == (Fig A6)
    # ------------------------------------------------------------------
    if "intermediate" in stages:
        dev = next(model.parameters()).device
        iid = H.tokenize_raw(tokenizer, P.INTERMEDIATE_PROMPT, dev)
        with torch.no_grad():
            out = model.generate(
                iid,
                attention_mask=torch.ones_like(iid),
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0, iid.shape[1] :], skip_special_tokens=True)
        log(f"intermediate completion: {completion!r} (paper: '27')")
        pg, aid, _ = P.build_graph_raw(
            model,
            tc,
            tokenizer,
            P.INTERMEDIATE_PROMPT,
            size_key=args.size,
            max_feature_nodes=args.graph_feature_nodes,
            out_html=str(out_dir / "graph_intermediate.html"),
            title="(4+5)*3",
        )
        nine_feats = [
            {
                "layer": n["layer"],
                "feature_idx": n["feature_idx"],
                "position": n["position"],
                "influence": n.get("influence", 0),
                "top_logits": (n.get("label") or {}).get("top_logits", [])[:5],
            }
            for n in pg["nodes"]
            if n["node_type"] == "feature"
            and any("9" in str(t) for t in ((n.get("label") or {}).get("top_logits") or [])[:5])
        ]
        nine_feats.sort(key=lambda d: -d["influence"])
        save_json(
            out_dir,
            "intermediate",
            {
                "completion": completion,
                "answer": tokenizer.decode([aid]),
                "nine_features": nine_feats[:15],
            },
        )

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    if "introspection" in stages:
        d = P.introspection_dialogue(model, tokenizer, A0, B0)
        log(f"introspection answer={d['answer']!r}")
        log(f"introspection explanation={d['explanation']!r}")
        save_json(out_dir, "introspection", d)

    # ------------------------------------------------------------------
    # corpus — dataset examples of lookup/sum features (Fig A3)
    # ------------------------------------------------------------------
    if "corpus" in stages:
        from datasets import load_dataset

        feats = [
            tuple(k) for k in state.get("lookup_feats", [])[:3] + state.get("sum_feats", [])[:3]
        ]
        if not feats:
            log("corpus: SKIPPED (no selected features; run grids first)")
        else:
            log(f"corpus: streaming {args.corpus_docs} C4 docs for {len(feats)} features")
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = []
            for ex in ds:
                t = ex["text"]
                if 200 < len(t) < 2000:
                    texts.append(t)
                if len(texts) >= args.corpus_docs:
                    break
            hits = P.corpus_scan(model, tc, feats, texts, tokenizer)
            save_json(
                out_dir,
                "corpus_examples",
                {f"L{L}f{i}": [(round(v, 2), s) for v, s in h] for (L, i), h in hits.items()},
            )

    save_json(out_dir, "state", state)
    log("suite done.")


if __name__ == "__main__":
    main()
