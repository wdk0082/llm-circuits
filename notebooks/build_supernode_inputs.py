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
  Addition: ones/first (studied pair), donor99 (49+49 ones), polymer (bare prompt).
* ``grids.npz`` — operand-grid evidence for EVERY feature node of the calc graphs
  (single readout pass per probe, cost ~10k forwards each, independent of feature
  count): ``first`` probe (peak over positions — input-side evidence) for all
  features; ``ones`` probe (teacher-forced final) for ones+donor final-position
  features; ``last`` probe for first-graph final-position features.
* ``polymer_ones_acts.json`` — activations of ones/donor final-position features at
  the polymer ONES moment (``POLYMER_PROMPT + "99"``), for the polymer-reuse
  supernode.
* ``manifest.json`` — exact prompts, token lists, key positions (operand tokens,
  digit positions, finals), answers, git sha.

Run (A100-80GB, warm caches, ~35 min):
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

A0, B0 = 46, 49  # the studied pair (auto-selected in earlier sessions; fixed here)
DA, DB = 49, 49  # the verified lookup(9,9) donor pair


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
        "--node-threshold",
        type=float,
        default=0.8,
        help="graph pruning node threshold for the dumps (recorded per graph in the"
        " manifest). The multilingual review pages use 0.95: the paper's supernode"
        " membership is activity-based (20/27 active vs 10/27 in its pruned graphs),"
        " and the explorer-export workflow can only pick GRAPH nodes, so selection"
        " needs the larger graph; notebooks re-prune in-memory where they need 0.8.",
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

    # ---- addition graphs --------------------------------------------------------------
    if not args.skip_addition:
        print("Addition graphs ...", flush=True)
        add_graphs: dict[str, dict] = {}
        for name, (a, b, target) in {
            "ones": (A0, B0, "ones"),
            "first": (A0, B0, "first"),
            "donor99": (DA, DB, "ones"),
        }.items():
            gd, ans_id, ids = A.build_addition_graph(
                model,
                tc,
                tokenizer,
                a,
                b,
                target=target,
                style="calc",
                size_key=size,
                node_threshold=args.node_threshold,
            )
            add_graphs[name] = gd
            dump_graph(out, name, gd)
            manifest["graphs"][name] = {
                "prompt": A.addition_prompt(a, b, "calc"),
                "pair": [a, b],
                "node_threshold": args.node_threshold,
                "target": target,
                "answer": tokenizer.decode([ans_id]),
                "answer_token_id": int(ans_id),
                "n_tokens": int(ids.shape[1]),
                "final_position": int(ids.shape[1] - 1),
                "digit_positions": A.digit_token_positions(tokenizer, ids, a, b),
            }
        pg, _p_aid, _ = A.build_graph_raw(
            model,
            tc,
            tokenizer,
            A.POLYMER_PROMPT,
            size_key=size,
            node_threshold=args.node_threshold,
        )
        dump_graph(out, "polymer", pg)
        manifest["graphs"]["polymer"] = {
            "prompt": A.POLYMER_PROMPT,
            "raw": True,
            "ones_moment_prompt": A.POLYMER_PROMPT + "99",
        }

        # ---- operand-grid evidence: one readout pass per probe, ALL features ----------
        a_vals, b_vals = list(range(100)), list(range(100))

        def feats_of(name: str, position: int | None = None) -> list[tuple[int, int]]:
            gd = add_graphs[name]
            return sorted(
                {
                    (n["layer"], n["feature_idx"])
                    for n in gd["nodes"]
                    if n["node_type"] == "feature"
                    and (position is None or n["position"] == position)
                }
            )

        fin_ones = manifest["graphs"]["ones"]["final_position"]
        fin_first = manifest["graphs"]["first"]["final_position"]
        fin_donor = manifest["graphs"]["donor99"]["final_position"]
        probe_sets = {
            "first": sorted(set(feats_of("ones")) | set(feats_of("first"))),
            "ones": sorted(set(feats_of("ones", fin_ones)) | set(feats_of("donor99", fin_donor))),
            "last": feats_of("first", fin_first),
        }
        grid_arrays: dict[str, np.ndarray] = {}
        for probe, feats in probe_sets.items():
            print(f"Grid pass probe={probe!r}: {len(feats)} features ...", flush=True)
            grids = A.feature_grids(
                model, tc, feats, a_vals, b_vals, tokenizer, probe=probe, style="calc"
            )
            grid_arrays[f"{probe}_grids"] = np.stack([grids[f] for f in feats])
            grid_arrays[f"{probe}_features"] = np.array(feats, dtype=np.int64)
            del grids
        np.savez_compressed(
            out / "grids.npz", a_vals=np.array(a_vals), b_vals=np.array(b_vals), **grid_arrays
        )
        print(f"  grids.npz: {(out / 'grids.npz').stat().st_size / 1e6:.1f} MB")

        # ---- polymer ones-moment activations for the reuse supernode -------------------
        probe_feats = probe_sets["ones"]
        acts, active_final, _ids = A.probe_features_on_prompt(
            model, tc, tokenizer, probe_feats, A.POLYMER_PROMPT + "99"
        )
        (out / "polymer_ones_acts.json").write_text(
            json.dumps(
                {
                    "prompt": A.POLYMER_PROMPT + "99",
                    "acts": acts,
                    "active_final": [[L, i, a] for (L, i, a) in active_final],
                },
                ensure_ascii=False,
                default=str,
            )
        )
        print(f"  polymer_ones_acts.json: {len(active_final)} active at the ones moment")

    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
