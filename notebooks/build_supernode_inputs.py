"""Stage 1 (GPU) of the supernode pipeline: persist everything the semantic selector
needs, so selection + human review can run offline (CPU) and be re-audited later.

Outputs under ``artifacts/supernode_inputs/<size>/``:

* ``graph_<name>.json`` — the PRUNED attribution-graph dicts (``graph_to_dict``
  output: nodes with layer/position/feature_idx/activation/influence and labels
  incl. activation examples, edges, tokens, prompt). These dicts were previously
  in-memory only (the explorer HTML drops influence).
  Multilingual: antonym_{en,fr,zh}, synonym_en, hot_en + raw_ variants of ALL of them.
  The raw forms are the paper's exact prompts (the paper never uses chat formatting);
  the chat forms are the Qwen3-instruct adaptation the notebook uses as primary.
  Addition (v3): ``first``/``ones`` (the studied pair's two answer moments), ``donor``
  (the _9+_9-class pair's ones moment — the paper's substitution donor), ``reuse``
  (the citation-style prose prompt at its ones moment). The studied pair is DERIVED
  here (accuracy grid -> ``pick_studied_pair``) and recorded in the manifest;
  re-runs reuse the recorded pair unless ``--repick-pair``.
* ``grids.npz`` — operand-grid evidence (the paper's operand plots) for the UNION of
  feature nodes across all four addition graphs, under all three probes
  (``final`` = the ``=`` token, ``ones`` = the teacher-forced ones moment,
  ``peak`` = max over positions). One readout pass per probe (~10k forwards each,
  independent of feature count); every review-page node gets its grids.
* ``reuse_moment_acts.json`` — calc-graph feature activations at the reuse prompt's
  ones moment (evidence that the lookup features fire in prose).
* ``manifest.json`` — exact prompts, token lists, key positions (operand tokens,
  digit positions, finals), answers, accuracy, pairs, git sha; addition entries
  carry ``"task": "addition"`` (the export-ingest routing key).

Run (A100-80GB, warm caches; sbatch hpc/run_supernode_inputs.sbatch):
    uv run python notebooks/build_supernode_inputs.py --size 4b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import addition_helper as A
import multilingual_helper as M
import numpy as np
import torch

from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder


def dump_graph(out_dir: Path, name: str, gd: dict) -> None:
    path = out_dir / f"graph_{name}.json"
    path.write_text(json.dumps(gd, ensure_ascii=False, default=str))
    n_feat = sum(1 for n in gd["nodes"] if n["node_type"] == "feature")
    print(f"  {path.name}: {n_feat} feature nodes, {path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--skip-multilingual", action="store_true")
    ap.add_argument("--skip-addition", action="store_true")
    ap.add_argument(
        "--graphs",
        nargs="*",
        default=None,
        help="build only these multilingual graph names (partial rebuild; the manifest"
        " merges, other entries stay). Addition graphs are all-or-nothing.",
    )
    ap.add_argument(
        "--repick-pair",
        action="store_true",
        help="re-derive the studied/donor pairs from a fresh accuracy grid even when the"
        " manifest already records them",
    )
    ap.add_argument(
        "--node-threshold",
        type=float,
        default=0.8,
        help="graph pruning node threshold for the dumps (recorded per graph in the"
        " manifest). Plan: CHAT multilingual graphs at 0.8 (circuit-tracer default),"
        " RAW graphs at 0.95 — the raw graphs starve at 0.8 (no quote-position nodes"
        " for the detectors) and the explorer-export workflow can only pick GRAPH"
        " nodes; the paper's own membership is activity-based (20/27 active vs 10/27"
        " in its pruned graphs). Rebuild one side via --graphs.",
    )
    args = ap.parse_args()
    size = args.size

    device = default_device()
    out = artifacts_dir() / "supernode_inputs" / size
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)

    print(f"Loading Qwen3-{size} + eager transcoders on {device} ...", flush=True)
    model, tokenizer = load_qwen3(size, dtype_str="bf16", device_map=device)
    model.eval()
    tc = load_transcoder(
        f"qwen3-{size}", device=device, dtype=torch.bfloat16, lazy_decoder=False
    ).transcoder

    # MERGE with an existing manifest: a partial rebuild (e.g. --skip-addition at a
    # different threshold) must not clobber the other task's entries.
    mpath = out / "manifest.json"
    manifest: dict = (
        json.loads(mpath.read_text()) if mpath.exists() else {"size": size, "graphs": {}}
    )
    manifest["size"] = size
    manifest["git"] = sha.stdout.strip()

    # ---- multilingual graphs (labels + examples embedded by build_graph) -------------
    if not args.skip_multilingual:
        print("Multilingual graphs ...", flush=True)
        ml_prompts: dict[str, tuple[str, bool]] = {}
        for lg in M.LANGS:
            ml_prompts[f"antonym_{lg}"] = (M.antonym_prompt("small", lg), False)
            ml_prompts[f"raw_antonym_{lg}"] = (M.raw_antonym_prompt("small", lg), True)
        ml_prompts["synonym_en"] = (M.synonym_prompt("small", "en"), False)
        ml_prompts["raw_synonym_en"] = (M.raw_synonym_prompt("small", "en"), True)
        ml_prompts["hot_en"] = (M.antonym_prompt("hot", "en"), False)
        ml_prompts["raw_hot_en"] = (M.raw_antonym_prompt("hot", "en"), True)
        for name, (prompt, raw) in ml_prompts.items():
            if args.graphs and name not in args.graphs:
                continue
            gd, ans_id, ids = M.build_graph(
                model,
                tc,
                tokenizer,
                prompt,
                size_key=size,
                raw=raw,
                node_threshold=args.node_threshold,
            )
            dump_graph(out, name, gd)
            operand = (
                "hot" if name in ("hot_en", "raw_hot_en") else M.WORD["small"][name.split("_")[-1]]
            )
            manifest["graphs"][name] = {
                "prompt": prompt,
                "raw": raw,
                "node_threshold": args.node_threshold,
                "answer": tokenizer.decode([ans_id]),
                "answer_token_id": int(ans_id),
                "n_tokens": int(ids.shape[1]),
                "final_position": int(ids.shape[1] - 1),
                "operand_position": M.operand_token_pos(tokenizer, ids, operand),
            }
            del gd
            torch.cuda.empty_cache()

    # ---- addition (v3): behavior -> pair, four graphs, three-probe grids ---------------
    if not args.skip_addition:
        recorded = manifest.get("addition") or {}
        if recorded.get("pair") and not args.repick_pair:
            (a0, b0), (da, db) = recorded["pair"], recorded["donor_pair"]
            note = recorded.get("pair_note", "") + " (pair reused from manifest)"
            acc_pct = recorded.get("accuracy_pct")
        else:
            print("Addition behavior: accuracy over the full operand grid ...", flush=True)
            correct, predicted = A.accuracy_grid(model, tokenizer)
            (a0, b0), note = A.pick_studied_pair(correct)
            donor = A.pick_pair_by_class(correct, ones=(9, 9), near=(a0, b0))
            if donor is None:
                raise SystemExit("no correct _9+_9 pair — cannot stage the paper's lookup swap")
            da, db = donor
            acc_pct = round(float(correct.mean()) * 100, 2)
            np.save(out / "accuracy_correct.npy", correct)
            np.save(out / "accuracy_predicted.npy", predicted)
        print(f"  pair {a0}+{b0} ({note}); donor {da}+{db}; accuracy {acc_pct}%", flush=True)
        manifest["addition"] = {
            "pair": [a0, b0],
            "donor_pair": [da, db],
            "pair_note": note,
            "accuracy_pct": acc_pct,
        }

        print("Addition graphs ...", flush=True)
        add_graphs: dict[str, dict] = {}
        for name, (pa, pb, target) in {
            "first": (a0, b0, "first"),
            "ones": (a0, b0, "ones"),
            "donor": (da, db, "ones"),
        }.items():
            gd, ans_id, ids = A.build_calc_graph(
                model,
                tc,
                tokenizer,
                pa,
                pb,
                target=target,
                size_key=size,
                node_threshold=args.node_threshold,
            )
            add_graphs[name] = gd
            dump_graph(out, name, gd)
            manifest["graphs"][name] = {
                "task": "addition",
                "prompt": A.addition_prompt(pa, pb, str(pa + pb)[:-1] if target == "ones" else ""),
                "pair": [pa, pb],
                "target": target,
                "node_threshold": args.node_threshold,
                "answer": tokenizer.decode([ans_id]),
                "answer_token_id": int(ans_id),
                "n_tokens": int(ids.shape[1]),
                "final_position": int(ids.shape[1] - 1),
                "digit_positions": A.digit_token_positions(tokenizer, ids, pa, pb),
            }
            del gd
            torch.cuda.empty_cache()

        # The citation-style reuse prompt at ITS ones moment (paper: the same lookup
        # features act in prose). Behavior is recorded, not required — a miss is a
        # documented negative, and the graph still shows what the model does instead.
        reuse_ones_prompt = A.reuse_prompt(a0, b0) + str(a0 + b0)[:-1]
        expected_ones = str(a0 + b0)[-1]
        gd, ans_id, ids = A.build_text_graph(
            model,
            tc,
            tokenizer,
            reuse_ones_prompt,
            size_key=size,
            node_threshold=args.node_threshold,
            metadata={"pair": [a0, b0], "target": "reuse-ones"},
        )
        add_graphs["reuse"] = gd
        dump_graph(out, "reuse", gd)
        manifest["graphs"]["reuse"] = {
            "task": "addition",
            "prompt": reuse_ones_prompt,
            "pair": [a0, b0],
            "target": "reuse-ones",
            "node_threshold": args.node_threshold,
            "answer": tokenizer.decode([ans_id]),
            "answer_token_id": int(ans_id),
            "expected_answer": expected_ones,
            "behavior_ok": tokenizer.decode([ans_id]).strip() == expected_ones,
            "n_tokens": int(ids.shape[1]),
            "final_position": int(ids.shape[1] - 1),
        }
        del gd
        torch.cuda.empty_cache()

        # ---- operand-grid evidence: the union of ALL four graphs' features under ALL
        # three probes (cost per probe ~10k forwards, independent of feature count) —
        # every review-page node gets its operand plots.
        a_vals, b_vals = list(range(100)), list(range(100))
        union = sorted(
            {
                (n["layer"], n["feature_idx"])
                for g in add_graphs.values()
                for n in g["nodes"]
                if n["node_type"] == "feature"
            }
        )
        grid_arrays: dict[str, np.ndarray] = {}
        for probe in ("final", "ones", "peak"):
            print(f"Grid pass probe={probe!r}: {len(union)} features ...", flush=True)
            grids = A.operand_grids(model, tc, tokenizer, union, a_vals, b_vals, probe=probe)
            grid_arrays[f"{probe}_grids"] = np.stack([grids[f] for f in union])
            grid_arrays[f"{probe}_features"] = np.array(union, dtype=np.int64)
            del grids
        np.savez_compressed(
            out / "grids.npz", a_vals=np.array(a_vals), b_vals=np.array(b_vals), **grid_arrays
        )
        print(f"  grids.npz: {(out / 'grids.npz').stat().st_size / 1e6:.1f} MB")

        # ---- reuse-moment activations (which calc features fire in the prose context) --
        _, active_final, _ = A.probe_features_on_prompt(
            model, tc, tokenizer, union, reuse_ones_prompt
        )
        (out / "reuse_moment_acts.json").write_text(
            json.dumps(
                {
                    "prompt": reuse_ones_prompt,
                    "active_final": [[L, i, act] for (L, i, act) in active_final],
                },
                ensure_ascii=False,
            )
        )
        print(f"  reuse_moment_acts.json: {len(active_final)} active at the reuse ones moment")

    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
