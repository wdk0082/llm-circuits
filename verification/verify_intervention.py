#!/usr/bin/env python3
"""Cross-verify our steering against circuit-tracer's ``feature_intervention``.

This is the gold-standard faithfulness check for ``mode="steering-base-model"``:
it runs **our** intervention (HF model + ``run_feature_intervention``) and
**circuit-tracer's** ``ReplacementModel.feature_intervention`` (TransformerLens) on
the *same* Qwen3 model, the *same* transcoders, the *same* prompt/features/values,
with matching constrained layer ranges, then compares the steering effect on the
logits.  Three cases cover every intervention primitive the reproductions use:

  A. single-feature negative steer (m=-2), constrained to its own layer — the
     original parity case;
  B. multi-layer suppression stack (two coupled features, m=-2) with the patch end
     layer ABOVE the top steered layer — exercises clean-anchored deltas for
     multiple features, the no-second-order-effects pin inside the range, and the
     within-range attention response (frozen patterns, live V) riding above the
     steers;
  C. ``value=`` donor injection — a feature (near-)inactive at the target position
     set to an absolute positive value taken from another position (the swap
     protocol's donor half), under the same constrained patching.

It lives OUTSIDE ``src/`` on purpose: the package uses circuit-tracer only to load
transcoders; importing ``ReplacementModel`` is confined to this verification folder.

The two sides are different implementations (HF vs TransformerLens), so their *clean*
logits differ by a small floor; we therefore compare the **steering delta**
(steered - clean) on each side, which cancels that baseline offset and isolates the
intervention method.  PASS = the delta match is within a few x the clean noise floor.

Run on a GPU node:
    uv run python verification/verify_intervention.py
"""

from __future__ import annotations

import sys

import torch

from llm_circuits.circuits.interventions import FeatureIntervention, run_feature_intervention
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

# Small model: this checks intervention *method* faithfulness (not transcoder quality),
# so 0.6B keeps it fast and lets both models + transcoders fit comfortably in memory.
MODEL_SIZE = "0.6b"
TL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
M = -2.0  # additive-delta multiple: -2 = negative steer (flip the feature's sign)


def compare_case(name, model, tc, rm, input_ids, ours_ivs, ct_ivs, patch_end_layer):
    """Run one intervention on both sides; return (ok, summary line)."""
    last = input_ids.shape[1] - 1

    ours = run_feature_intervention(
        model,
        tc,
        input_ids,
        ours_ivs,
        patch_end_layer=patch_end_layer,
        freeze_attention=True,
        n_bos_tokens=1,
    )
    our_delta = (ours.ablated_logits[last] - ours.baseline_logits[last]).float()
    our_clean_logits = ours.baseline_logits[last].float()

    ct_clean_logits, _ = rm.feature_intervention(input_ids, [], freeze_attention=True)
    ct_steer_logits, _ = rm.feature_intervention(
        input_ids,
        ct_ivs,
        constrained_layers=range(0, patch_end_layer + 1),
        freeze_attention=True,
    )
    ct_clean_logits = ct_clean_logits[0, last].float()
    ct_delta = (ct_steer_logits[0, last] - ct_clean_logits).float()

    clean_noise = (our_clean_logits - ct_clean_logits).abs().max().item()
    delta_diff = (our_delta - ct_delta).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(our_delta, ct_delta, dim=0).item()
    tol = max(5.0 * clean_noise, 1e-2)
    ok = delta_diff <= tol and cos > 0.99

    print(f"\n--- case {name} ---")
    print(f"  noise floor (HF vs TL clean logits):   {clean_noise:.5f}")
    print(f"  ||our_delta||={our_delta.norm():.3f}  ||ct_delta||={ct_delta.norm():.3f}")
    print(f"  max |our_delta - ct_delta|:            {delta_diff:.5f}  (tol {tol:.5f})")
    print(f"  cosine(our_delta, ct_delta):           {cos:.5f}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok, f"{name}: {'PASS' if ok else 'FAIL'} (dw {delta_diff:.5f}, cos {cos:.5f})"


def main() -> None:
    device = default_device()
    # fp32: a faithfulness check needs the HF-vs-TransformerLens numerical floor to be
    # tiny (bf16 rounding ~0.3 logits swamps the steering signal).
    dtype = torch.float32
    dtype_str = "fp32"

    # circuit-tracer's TransformerLens ReplacementModel (confined to this folder).
    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )

    # Our HF model + the shared transcoder set.
    print(f"Loading our HF Qwen3-{MODEL_SIZE} + transcoders ...", flush=True)
    model, _ = load_qwen3(MODEL_SIZE, dtype_str=dtype_str, device_map=device)
    model.eval()
    tc = load_transcoder(
        f"qwen3-{MODEL_SIZE}", device=device, dtype=dtype, lazy_decoder=False
    ).transcoder

    # circuit-tracer ReplacementModel reusing the SAME transcoder object.
    print(f"Loading circuit-tracer ReplacementModel ({TL_NAME}) ...", flush=True)
    rm = TransformerLensReplacementModel.from_pretrained_and_transcoders(
        TL_NAME, tc, device=device, dtype=dtype
    )

    # Shared tokens (use circuit-tracer's tokenizer so both sides see identical ids).
    input_ids = rm.to_tokens(PROMPT)  # (1, seq)
    last = input_ids.shape[1] - 1
    n_layers = len(tc)
    layer = n_layers // 2  # a mid-network layer

    # Clean feature activations on both sides (ours anchor our deltas; circuit-tracer's
    # anchor the absolute `value` targets its API takes, cancelling the tiny HF-vs-TL
    # activation noise from the comparison).
    clean = run_feature_intervention(model, tc, input_ids, [], n_bos_tokens=1)

    def our_acts(lyr):
        feats = clean.baseline_features[lyr]
        return feats[0] if feats.dim() == 3 else feats

    # get_activations returns (logits, activation_cache) — take the cache.
    ct_acts_out = rm.get_activations(input_ids)
    ct_acts = ct_acts_out[1] if isinstance(ct_acts_out, tuple) else ct_acts_out

    def ct_act(lyr, pos, fid):
        la = ct_acts[lyr]
        la = la[0] if la.dim() == 3 else la
        return float(la[pos, fid])

    results = []

    # --- A. single feature, m=-2, constrained to its own layer (the original case) ---
    fid = int(our_acts(layer)[last].abs().argmax())
    print(
        f"\ncase A feature: layer={layer} pos={last} fid={fid} "
        f"clean={float(our_acts(layer)[last, fid]):.4f}, M={M}"
    )
    results.append(
        compare_case(
            "A single-feature m=-2 @ell=layer",
            model,
            tc,
            rm,
            input_ids,
            [FeatureIntervention(layer, fid, position=last, m=M)],
            [(layer, last, fid, (1.0 + M) * ct_act(layer, last, fid))],
            patch_end_layer=layer,
        )
    )

    # --- B. multi-layer coupled stack, m=-2 x2, patch end ABOVE the top steer ---
    layer2 = layer + 2
    fid2 = int(our_acts(layer2)[last].abs().argmax())
    ell_b = min(layer2 + 3, n_layers - 1)
    print(
        f"\ncase B features: L{layer}f{fid} + L{layer2}f{fid2} "
        f"(clean {float(our_acts(layer2)[last, fid2]):.4f}), ell={ell_b}"
    )
    results.append(
        compare_case(
            f"B two-layer stack m=-2 @ell={ell_b}",
            model,
            tc,
            rm,
            input_ids,
            [
                FeatureIntervention(layer, fid, position=last, m=M),
                FeatureIntervention(layer2, fid2, position=last, m=M),
            ],
            [
                (layer, last, fid, (1.0 + M) * ct_act(layer, last, fid)),
                (layer2, last, fid2, (1.0 + M) * ct_act(layer2, last, fid2)),
            ],
            patch_end_layer=ell_b,
        )
    )

    # --- C. value= donor injection: take the layer's top feature at an EARLIER position
    # (the "donor") and inject 1.5x its activation at the last position, where it is
    # (near-)inactive — the swap protocol's donor half. ---
    donor_pos = max(1, last - 2)
    fid_d = int(our_acts(layer)[donor_pos].abs().argmax())
    donor_value = 1.5 * float(our_acts(layer)[donor_pos, fid_d])
    recip_clean = float(our_acts(layer)[last, fid_d])
    ell_c = min(layer + 3, n_layers - 1)
    print(
        f"\ncase C donor: L{layer}f{fid_d} act@p{donor_pos}={donor_value / 1.5:.4f} -> "
        f"inject value={donor_value:.4f} at p{last} (recipient clean {recip_clean:.4f}), "
        f"ell={ell_c}"
    )
    results.append(
        compare_case(
            f"C value= donor injection @ell={ell_c}",
            model,
            tc,
            rm,
            input_ids,
            [FeatureIntervention(layer, fid_d, position=last, value=donor_value)],
            [(layer, last, fid_d, donor_value)],
            patch_end_layer=ell_c,
        )
    )

    print("\n=== summary ===")
    for _ok, line in results:
        print(" ", line)
    if not all(ok for ok, _ in results):
        # a FAIL must be a failing exit code so this can gate regressions (DEVLOG_EXTRA 4.1)
        sys.exit(1)


if __name__ == "__main__":
    main()
