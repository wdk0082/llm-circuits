"""Polymer reuse probe: are the calc lookup/sum features ACTIVE on the citation prompt?

The suite's polymer stage tests reuse by pruned-graph membership, which is stricter than
the paper's claim (the feature is active and causally relevant).  This probe reads the raw
transcoder activations of the selected lookup/sum features (from the addition suite's
``state.json``) at every position of the polymer prompt, and — if any are active at the
final position — suppresses them at the paper's -2x (our m=-3) to test causality.

Usage: uv run python examples/paper_addition_polymer_probe.py --size 4b
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

from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import artifacts_dir, default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[probe +{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="4b")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    device = default_device()
    out_dir = artifacts_dir() / "paper_addition" / args.size
    state = json.loads((out_dir / "state.json").read_text())
    feats = [tuple(k) for k in state.get("lookup_feats", []) + state.get("sum_feats", [])]
    if not feats:
        raise SystemExit("no lookup/sum selections in state.json — run the grids stage first")

    model, tokenizer = load_qwen3(args.size, dtype_str=args.dtype, device_map=device)
    model.eval()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tc = load_transcoder(f"qwen3-{args.size}", device=device, dtype=dtype).transcoder
    log(f"loaded; probing {len(feats)} calc lookup/sum features on the polymer prompt")

    # Two probe points, mirroring the calc ones-circuit's teacher-forced probe:
    #   bare  — prompt ends "..., 1": predicting the FIRST digit '9' (magnitude moment);
    #   ones  — prompt + "99" (teacher-forced): predicting the final '5' of 1995 — the
    #           citation's ones-digit moment, where a _6+_9 -> _5 lookup should fire.
    dev = next(model.parameters()).device
    prompts = {"bare": P.POLYMER_PROMPT, "ones": P.POLYMER_PROMPT + "99"}
    ids_by = {k: H.tokenize_raw(tokenizer, p, dev) for k, p in prompts.items()}

    layers = sorted({L for L, _ in feats})
    result: dict = {"features": [f"L{L}f{f}" for L, f in feats]}

    for tag, ids in ids_by.items():
        toks = [tokenizer.decode([int(t)]) for t in ids[0]]
        captured: dict[int, torch.Tensor] = {}

        def mk(layer: int):
            def hook(_m, inp, _o):
                captured[layer] = inp[0]  # noqa: B023

            return hook

        handles = [
            model.get_submodule(f"model.layers.{L}.mlp").register_forward_hook(mk(L))
            for L in layers
        ]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in handles:
                h.remove()

        acts: dict[str, dict] = {}
        active_final: list[tuple[int, int, float]] = []
        last = ids.shape[1] - 1
        with torch.no_grad():
            for L in layers:
                enc = tc.transcoders[L].encode(captured[L])[0]  # (seq, d_t)
                for L2, f in feats:
                    if L2 != L:
                        continue
                    row = enc[:, f]
                    nz = [
                        (i, toks[i], round(float(row[i]), 2))
                        for i in range(1, len(toks))
                        if row[i] > 0
                    ]
                    acts[f"L{L}f{f}"] = {"nonzero": nz, "final": round(float(row[last]), 3)}
                    if float(row[last]) > 0:
                        active_final.append((L, f, float(row[last])))

        for k, v in acts.items():
            log(f"[{tag}] {k}: final={v['final']} nonzero@{[(i, t) for i, t, _ in v['nonzero']]}")

        entry: dict = {"prompt": prompts[tag], "tokens": toks, "activations": acts}
        if active_final:
            log(
                f"[{tag}] {len(active_final)} features active at final position -> "
                f"suppress at m=-3 (paper -2x)"
            )
            rep = P.steer_and_report(model, tc, ids, active_final, tokenizer, m=-3.0, position=last)
            entry["suppress_active_m-3"] = rep
            log(
                f"[{tag}] suppression: baseline_top={rep['baseline_top'][:3]} "
                f"steered_top={rep['steered_top'][:3]}"
            )
        else:
            log(f"[{tag}] NO calc lookup/sum feature is active at the final position")
        result[tag] = entry

        # Paper Fig A5 panel 3 — lookup SWAP: _6+_9 at -1x (our m=-2) plus _9+_9 at +1x
        # turned 995 into 998.  Our analog at the citation's ones moment: flip the active
        # (6,9)-class lookups and inject the (9,9) lookup at 1x the activation it has on a
        # (9,9) calc prompt (49+49=98 -> ones digit 8), expecting the prediction to move
        # toward '8' (1959+39=1998).
        if tag == "ones" and active_final:
            import json as _json

            taxonomy = _json.loads((out_dir / "taxonomy.json").read_text())
            donors = [
                tuple(int(x) for x in key[1:].split("f"))
                for grp in ("midlayer", "answer_ones")
                for key, rep in taxonomy.get(grp, {}).items()
                if str(rep.get("label", "")).startswith("lookup(a%10=9,b%10=9)")
            ]
            donors = sorted(set(donors))
            lookup_keys = {tuple(k) for k in state.get("lookup_feats", [])}
            active_lookups = [(L, i, a) for (L, i, a) in active_final if (L, i) in lookup_keys]
            if donors and active_lookups:
                ids99 = H.tokenize_addition(tokenizer, 49, 49, dev, style="calc")
                ans99 = tokenizer("98", add_special_tokens=False, return_tensors="pt").input_ids
                ids99 = torch.cat([ids99, ans99[:, :-1].to(dev)], dim=1)  # teacher-force '9'
                cap99: dict[int, torch.Tensor] = {}
                dlayers = sorted({L for L, _ in donors})
                handles = [
                    model.get_submodule(f"model.layers.{L}.mlp").register_forward_hook(mk(L))
                    for L in dlayers
                ]
                captured.clear()
                try:
                    with torch.no_grad():
                        model(ids99)
                        for L in dlayers:
                            cap99[L] = tc.transcoders[L].encode(captured[L])[0]
                finally:
                    for h in handles:
                        h.remove()
                donor_acts = [
                    (L, i, float(cap99[L][-1, i])) for (L, i) in donors if cap99[L][-1, i] > 0
                ]
                log(f"[swap] donors (9,9) act on 49+49-ones: {donor_acts}")
                if donor_acts:
                    from llm_circuits.circuits.interventions import (
                        FeatureIntervention,
                        run_feature_intervention,
                    )

                    last = ids.shape[1] - 1
                    ivs = [
                        FeatureIntervention(L, i, position=last, m=-2.0)
                        for (L, i, _) in active_lookups
                    ]
                    ivs += [
                        FeatureIntervention(L, i, position=last, value=a)
                        for (L, i, a) in donor_acts
                    ]
                    res = run_feature_intervention(model, tc, ids, ivs, n_bos_tokens=H.N_BOS)
                    swap_rep = {
                        "suppressed_lookups": active_lookups,
                        "injected_donors_1x": donor_acts,
                        "baseline_top": P.top_tokens(res.baseline_logits[-1], tokenizer),
                        "swapped_top": P.top_tokens(res.ablated_logits[-1], tokenizer),
                    }
                    result["lookup_swap_69_to_99"] = swap_rep
                    log(
                        f"[swap] baseline={swap_rep['baseline_top'][:3]} -> "
                        f"swapped={swap_rep['swapped_top'][:3]} (paper: 995->998)"
                    )

    (out_dir / "polymer_probe.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=str)
    )
    log(f"wrote {out_dir / 'polymer_probe.json'}")


if __name__ == "__main__":
    main()
