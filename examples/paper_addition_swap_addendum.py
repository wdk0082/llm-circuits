"""Paper Fig A5 panel 3 addendum — the lookup SWAP (_6+_9 -> _9+_9, 995 -> 998).

The probe (``paper_addition_polymer_probe.py``) reproduced Fig A5's suppression panel;
the swap panel needs a verified ``_9+_9``-class lookup donor, which the 46+49 selection
does not contain (the auto-labelled (9,9) candidate is really an ends-in-0/5 sum feature).

This script sources the donor honestly:
  1. build the ones-digit graph for ``calc: 49+49=`` (teacher-forced '9', predicting '8');
  2. take its influence-top mid-layer features at the final position;
  3. operand-grid-probe JUST those candidates and keep lookup-class ones
     (jointly residue-selective, the paper's point-lattice signature);
  4. on the polymer ones-moment prompt ("..., 199" -> '5'), flip the active (6,9)-class
     lookups (paper -1x == our m=-2) and inject the verified donors at 1x their 49+49
     activation; report the prediction shift (paper: 995 -> 998, i.e. expect '8').

If step 3 finds no lookup-class donor, that is recorded as the result: the per-layer
transcoder inventory lacks a usable _9+_9 lookup feature (a dictionary difference vs the
paper's CLT, not an intervention failure).

Usage: uv run python examples/paper_addition_swap_addendum.py --size 4b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import addition_paper as P
import helper as H

from llm_circuits.circuits.interventions import FeatureIntervention, run_feature_intervention
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

T0 = time.perf_counter()
A9, B9 = 49, 49  # the donor problem: 9+9 -> ones digit 8


def log(msg: str) -> None:
    print(f"[swap +{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="4b")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--graph-feature-nodes", type=int, default=8000)
    ap.add_argument("--candidates", type=int, default=12)
    args = ap.parse_args()

    device = default_device()
    out_dir = artifacts_dir() / "paper_addition" / args.size
    state = json.loads((out_dir / "state.json").read_text())
    lookup_keys = {tuple(k) for k in state.get("lookup_feats", [])}

    model, tokenizer = load_qwen3(args.size, dtype_str=args.dtype, device_map=device)
    model.eval()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tc = load_transcoder(f"qwen3-{args.size}", device=device, dtype=dtype).transcoder
    n_layers = len(tc)
    dev = next(model.parameters()).device
    log(f"loaded Qwen3-{args.size}; building donor graph for {A9}+{B9}")

    # 1. donor graph (ones digit of 49+49=98)
    g99, ans99, ids99 = H.build_addition_graph(
        model,
        tc,
        tokenizer,
        A9,
        B9,
        target="ones",
        style=state.get("style", "calc"),
        max_feature_nodes=args.graph_feature_nodes,
        out_html=str(out_dir / f"graph_{A9}p{B9}_ones.html"),
    )
    log(f"donor graph answer={tokenizer.decode([ans99])!r} (expect '8')")

    # 2. mid-layer candidates at the final position, influence-ranked
    last99 = ids99.shape[1] - 1
    mid_lo, mid_hi = n_layers // 4, 3 * n_layers // 4
    cands = [
        n
        for n in g99["nodes"]
        if n["node_type"] == "feature" and n["position"] == last99 and mid_lo <= n["layer"] < mid_hi
    ]
    cands.sort(key=lambda n: n.get("influence", 0.0), reverse=True)
    cands = cands[: args.candidates]
    cand_keys = [(n["layer"], n["feature_idx"]) for n in cands]
    cand_acts = {(n["layer"], n["feature_idx"]): float(n.get("activation", 0.0)) for n in cands}
    log(f"donor candidates: {[f'L{L}f{i}' for L, i in cand_keys]}")

    # 3. operand-grid probe of the candidates only; keep lookup-class features
    A_VALS = list(range(100))
    B_VALS = list(range(100))
    grids = H.feature_grids(
        model,
        tc,
        cand_keys,
        A_VALS,
        B_VALS,
        tokenizer,
        batch_size=256,
        probe="ones",
        style=state.get("style", "calc"),
    )
    donors: list[tuple[int, int, float]] = []
    taxonomy: dict[str, dict] = {}
    for L, i in cand_keys:
        rep = H.periodicity_report(grids[(L, i)], A_VALS, B_VALS)
        taxonomy[f"L{L}f{i}"] = rep
        if str(rep["label"]).startswith("lookup") and cand_acts[(L, i)] > 0:
            donors.append((L, i, cand_acts[(L, i)]))
        log(f"candidate L{L}f{i}: {rep['label']} (act {cand_acts[(L, i)]:.2f})")

    result: dict = {
        "donor_problem": f"{A9}+{B9}",
        "donor_answer": tokenizer.decode([ans99]),
        "candidate_taxonomy": taxonomy,
        "donors": donors,
    }

    # 4. the swap on the polymer ones-moment prompt
    if donors:
        ids = H.tokenize_raw(tokenizer, P.POLYMER_PROMPT + "99", dev)
        last = ids.shape[1] - 1
        probe = json.loads((out_dir / "polymer_probe.json").read_text())
        active = {k: v["final"] for k, v in probe["ones"]["activations"].items() if v["final"] > 0}
        active_lookups = [
            (L, i, active[f"L{L}f{i}"]) for (L, i) in lookup_keys if f"L{L}f{i}" in active
        ]
        log(f"suppressing {len(active_lookups)} active lookups, injecting {len(donors)} donors")
        ivs = [FeatureIntervention(L, i, position=last, m=-2.0) for (L, i, _) in active_lookups]
        ivs += [FeatureIntervention(L, i, position=last, value=a) for (L, i, a) in donors]
        res = run_feature_intervention(model, tc, ids, ivs, n_bos_tokens=H.N_BOS)
        result["swap"] = {
            "suppressed_lookups_m-2": active_lookups,
            "injected_donors_1x": donors,
            "baseline_top": P.top_tokens(res.baseline_logits[-1], tokenizer),
            "swapped_top": P.top_tokens(res.ablated_logits[-1], tokenizer),
        }
        log(
            f"swap: baseline={result['swap']['baseline_top'][:3]} -> "
            f"swapped={result['swap']['swapped_top'][:3]} (paper: 995->998, expect '8')"
        )
    else:
        result["swap"] = "no lookup-class donor found in the 49+49 mid-layer candidates"
        log("NO lookup-class donor found — swap not replicable with this dictionary")

    (out_dir / "lookup_swap.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=str)
    )
    log(f"wrote {out_dir / 'lookup_swap.json'}")


if __name__ == "__main__":
    main()
