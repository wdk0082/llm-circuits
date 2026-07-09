"""PAPER-EXACT multilingual reproduction suite (biology.html, Multilingual Circuits).

Runs every experiment in the paper's multilingual dive with the paper's protocols:

  behavior   — antonym + synonym answers in EN/FR/ZH (task competence, Fig B1 setup).
  graphs     — attribution graphs: antonym x3 languages, EN synonym, EN hot (donors).
  swap_operation — antonym -> synonym: source -5x, donor +6x, swept 0->6x with the
               crossover reported per language (paper: ~4x; Fig B3).
  swap_operand   — small -> hot: source -0.5x, donor +1.5x, swept (Fig B4).
  swap_language  — early language-detection features, the paper's three directions
               EN->ZH, FR->EN, ZH->FR at -5x/+6x, swept (Fig B5).
  overlap    — the paper's cross-lingual IOU (features active anywhere in context,
               translated paragraphs, UNRELATED-pair baseline; Fig B7). Run this stage
               at two sizes (0.6b and 4b) to reproduce the scale comparison.

Artifacts (JSON + PNG + explorer HTML) land in
``$LLM_CIRCUITS_ARTIFACTS_DIR/paper_multilingual/<size>/``.

Usage (locally or on the TPU via ``gcp/launch.sh``):
    uv run python examples/paper_multilingual_suite.py --size 4b --stages all
    gcp/launch.sh examples/paper_multilingual_suite.py --size 0.6b --stages behavior,overlap
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import multilingual_helper as M

from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

ALL_STAGES = ["behavior", "graphs", "swap_operation", "swap_operand", "swap_language", "overlap"]


def log(msg: str) -> None:
    print(f"[suite +{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


T0 = time.perf_counter()


def save_json(out_dir: Path, name: str, obj) -> None:
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    log(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="4b", help="Qwen3 size key (0.6b, 1.7b, 4b, ...)")
    ap.add_argument("--stages", default="all", help="comma list or 'all'")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--graph-feature-nodes", type=int, default=8000)
    ap.add_argument("--sweep-steps", type=int, default=13)
    args = ap.parse_args()
    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")

    device = default_device()
    out_dir = artifacts_dir() / "paper_multilingual" / args.size
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"device={device} size={args.size} stages={stages} -> {out_dir}")

    model, tokenizer = load_qwen3(args.size, dtype_str=args.dtype, device_map=device)
    model.eval()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tc = load_transcoder(f"qwen3-{args.size}", device=device, dtype=dtype).transcoder
    log(f"loaded Qwen3-{args.size} + transcoders ({len(tc)} layers)")

    # ------------------------------------------------------------------
    # behavior — task answers per language (baseline + expected swap targets)
    # ------------------------------------------------------------------
    behavior: dict = {"antonym": {}, "synonym": {}, "antonym_hot": {}}
    if "behavior" in stages or any(s.startswith("swap") for s in stages):
        for lg in M.LANGS:
            for key, prompt in [
                ("antonym", M.antonym_prompt("small", lg)),
                ("synonym", M.synonym_prompt("small", lg)),
                ("antonym_hot", M.antonym_prompt("hot", lg)),
            ]:
                tid, tstr, cont = M.model_answer(model, tokenizer, prompt)
                behavior[key][lg] = {"token_id": tid, "token": tstr, "continuation": cont}
                log(f"behavior {key:12s} {lg}: {tstr!r} ({cont!r})")
        save_json(out_dir, "behavior", behavior)

    # ------------------------------------------------------------------
    # graphs — antonym x3 + EN synonym + EN hot
    # ------------------------------------------------------------------
    graphs: dict = {}
    answers: dict = {}
    ids: dict = {}
    if "graphs" in stages or any(s.startswith("swap") for s in stages):
        specs = [(f"antonym_{lg}", M.antonym_prompt("small", lg)) for lg in M.LANGS]
        specs += [("synonym_en", M.synonym_prompt("small", "en"))]
        specs += [("hot_en", M.antonym_prompt("hot", "en"))]
        for name, prompt in specs:
            t = time.perf_counter()
            d, aid, iid = M.build_graph(
                model,
                tc,
                tokenizer,
                prompt,
                max_feature_nodes=args.graph_feature_nodes,
                out_html=str(out_dir / f"graph_{name}.html"),
                title=name,
            )
            graphs[name], answers[name], ids[name] = d, aid, iid
            log(
                f"graph {name}: answer={tokenizer.decode([aid])!r} ({time.perf_counter() - t:.0f}s)"
            )
        save_json(
            out_dir,
            "graph_answers",
            {k: {"answer": tokenizer.decode([v]), "token_id": v} for k, v in answers.items()},
        )

    # ------------------------------------------------------------------
    # swap stages (paper protocol + sweep + crossover)
    # ------------------------------------------------------------------
    def run_swaps(kind: str, jobs: list[dict]) -> None:
        results, token_strs = {}, {}
        for job in jobs:
            lg = job["lang"]
            r = M.paper_swap_sweep(
                model,
                tc,
                job["recipient_ids"],
                job["source"],
                job["donor"],
                job["position"],
                tokenizer,
                kind=kind,
                baseline_token=job["baseline_token"],
                expected_token=job["expected_token"],
                n_steps=args.sweep_steps,
            )
            r["label"] = job["label"]
            results[lg] = r
            token_strs[lg] = (
                tokenizer.decode([job["baseline_token"]]).strip(),
                tokenizer.decode([job["expected_token"]]).strip(),
            )
            log(
                f"{kind} {job['label']}: crossover={r['crossover']} "
                f"p_exp(max)={max(r['p_expected']):.3f} final_tops={r['top_tokens'][-1][:2]}"
            )
        M.plt.figure()
        M.plot_swap_sweeps(results, tokenizer, token_strs, title=f"{kind} (paper protocol)")
        M.plt.savefig(out_dir / f"{kind}_sweep.png", dpi=130, bbox_inches="tight")
        M.plt.close("all")
        save_json(out_dir, f"{kind}_results", results)

    if "swap_operation" in stages:
        donor_syn = M.graph_features_at_position(
            graphs["synonym_en"], ids["synonym_en"].shape[1] - 1, top_n=10
        )
        jobs = []
        for lg in M.LANGS:
            g = graphs[f"antonym_{lg}"]
            src = M.graph_features_at_position(g, ids[f"antonym_{lg}"].shape[1] - 1, top_n=10)
            jobs.append(
                {
                    "lang": lg,
                    "label": f"{lg}: antonym->synonym",
                    "recipient_ids": ids[f"antonym_{lg}"],
                    "source": src,
                    "donor": donor_syn,
                    "position": "final",
                    "baseline_token": behavior["antonym"][lg]["token_id"],
                    "expected_token": behavior["synonym"][lg]["token_id"],
                }
            )
        run_swaps("operation", jobs)

    if "swap_operand" in stages:
        hot_pos = M.operand_token_pos(tokenizer, ids["hot_en"], "hot")
        donor_hot = M.graph_features_at_position(graphs["hot_en"], hot_pos, top_n=10)
        jobs = []
        for lg in M.LANGS:
            g = graphs[f"antonym_{lg}"]
            spos = M.operand_token_pos(tokenizer, ids[f"antonym_{lg}"], M.WORD["small"][lg])
            src = M.graph_features_at_position(g, spos, top_n=10)
            jobs.append(
                {
                    "lang": lg,
                    "label": f"{lg}: small->hot",
                    "recipient_ids": ids[f"antonym_{lg}"],
                    "source": src,
                    "donor": donor_hot,
                    "position": spos,
                    "baseline_token": behavior["antonym"][lg]["token_id"],
                    "expected_token": behavior["antonym_hot"][lg]["token_id"],
                }
            )
        run_swaps("operand", jobs)

    if "swap_language" in stages:
        spec = M.early_language_detection_supernode(model, tc, tokenizer, "small")
        save_json(
            out_dir,
            "language_detection_supernodes",
            {lg: [(L, i, a) for (L, i, a) in v] for lg, v in spec.items()},
        )
        # The paper's three directions (Fig B5): EN->ZH, FR->EN, ZH->FR.
        directions = [("en", "zh"), ("fr", "en"), ("zh", "fr")]
        jobs = []
        for src_lg, tgt_lg in directions:
            jobs.append(
                {
                    "lang": src_lg,
                    "label": f"{src_lg}->{tgt_lg}",
                    "recipient_ids": ids[f"antonym_{src_lg}"],
                    "source": spec[src_lg],
                    "donor": spec[tgt_lg],
                    "position": "final",
                    "baseline_token": behavior["antonym"][src_lg]["token_id"],
                    "expected_token": behavior["antonym"][tgt_lg]["token_id"],
                }
            )
        run_swaps("language", jobs)

    # ------------------------------------------------------------------
    # overlap — the paper's IOU protocol with baseline (run per size)
    # ------------------------------------------------------------------
    if "overlap" in stages:
        corpus = M.CORPUS + M.PARAGRAPHS
        log(f"overlap: {len(corpus)} parallel items (paper IOU + unrelated baseline)")
        curves = M.overlap_curves_paper(model, tc, tokenizer, corpus)
        save_json(out_dir, "overlap_curves", {k: v.tolist() for k, v in curves.items()})
        fig, ax = M.plt.subplots(figsize=(8, 4.5))
        for key, arr in curves.items():
            if key.endswith("baseline"):
                ax.plot(range(len(arr)), arr, "--", alpha=0.5, label=key)
            else:
                ax.plot(range(len(arr)), arr, lw=2.2 if key == "mean" else 1.3, label=key)
        ax.set_xlabel("layer")
        ax.set_ylabel("IOU of active-feature sets")
        ax.set_title(f"Cross-lingual overlap, Qwen3-{args.size} (paper protocol + baseline)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "overlap_paper.png", dpi=130)
        M.plt.close("all")

    log("suite done.")


if __name__ == "__main__":
    main()
