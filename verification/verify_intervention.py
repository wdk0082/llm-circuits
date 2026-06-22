#!/usr/bin/env python3
"""Cross-verify our steering against circuit-tracer's ``feature_intervention``.

This is the gold-standard faithfulness check for ``mode="steering-base-model"``:
it runs **our** intervention (HF model + ``run_feature_intervention``) and
**circuit-tracer's** ``ReplacementModel.feature_intervention`` (TransformerLens) on
the *same* Qwen3 model, the *same* transcoders, the *same* prompt/feature/M, and a
matching constrained layer range, then compares the steering effect on the logits.

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

    # Pick the most active feature at the last position in that layer (our clean acts).
    clean = run_feature_intervention(model, tc, input_ids, [], n_bos_tokens=1)
    feats = clean.baseline_features[layer]
    if feats.dim() == 3:
        feats = feats[0]
    fid = int(feats[last].abs().argmax())
    our_clean_act = float(feats[last, fid])
    print(f"\nfeature: layer={layer} pos={last} fid={fid} clean_act={our_clean_act:.4f}, M={M}")

    # circuit-tracer's clean activation for the same feature (its delta basis). Best-effort:
    # if the return shape surprises us, fall back to our clean act (reported either way).
    ct_clean_act = None
    try:
        out = rm.get_activations(input_ids)
        acts = out[0] if isinstance(out, tuple) else out
        layer_acts = acts[layer]
        if layer_acts.dim() == 3:
            layer_acts = layer_acts[0]
        ct_clean_act = float(layer_acts[last, fid])
    except Exception as exc:
        print(f"  (could not read circuit-tracer clean act: {exc}; using ours as basis)")

    # --- ours: steering-base-model, constrained to [0, layer] ---
    ours = run_feature_intervention(
        model,
        tc,
        input_ids,
        [FeatureIntervention(layer, fid, position=last, m=M)],
        mode="steering-base-model",
        patch_end_layer=layer,
        freeze_attention=True,
        n_bos_tokens=1,
    )
    our_delta = (ours.ablated_logits[last] - ours.baseline_logits[last]).float()
    our_clean_logits = ours.baseline_logits[last].float()

    # --- circuit-tracer: feature_intervention with the matching constrained range ---
    basis = ct_clean_act if ct_clean_act is not None else our_clean_act
    value = (1.0 + M) * basis  # absolute target activation (their API takes a value)
    ct_clean_logits, _ = rm.feature_intervention(input_ids, [], freeze_attention=True)
    ct_steer_logits, _ = rm.feature_intervention(
        input_ids,
        [(layer, last, fid, value)],
        constrained_layers=range(0, layer + 1),
        freeze_attention=True,
    )
    ct_clean_logits = ct_clean_logits[0, last].float()
    ct_delta = (ct_steer_logits[0, last] - ct_clean_logits).float()

    # --- report ---
    clean_noise = (our_clean_logits - ct_clean_logits).abs().max().item()
    delta_diff = (our_delta - ct_delta).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(our_delta, ct_delta, dim=0).item()
    print(f"\nclean-logit noise floor (HF vs TransformerLens):  {clean_noise:.5f}")
    if ct_clean_act is not None:
        print(f"clean-act diff (ours {our_clean_act:.4f} vs ct {ct_clean_act:.4f})")
    print(
        f"steering magnitude ||our_delta||={our_delta.norm():.3f} ||ct_delta||={ct_delta.norm():.3f}"
    )
    print(f"max |our_delta - ct_delta| over vocab:            {delta_diff:.5f}")
    print(f"cosine(our_delta, ct_delta):                      {cos:.5f}")

    # PASS if the steering effects agree to within a few x the implementation noise floor.
    tol = max(5.0 * clean_noise, 1e-2)
    ok = delta_diff <= tol and cos > 0.99
    print(
        f"\n{'PASS' if ok else 'FAIL'}: delta_diff {delta_diff:.5f} <= tol {tol:.5f} and cos>0.99"
    )


if __name__ == "__main__":
    main()
