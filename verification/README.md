# verification/

Cross-checks of our from-scratch implementations against **circuit-tracer** itself.

These scripts are intentionally kept **outside `src/`**: the package
(`src/llm_circuits/`) uses circuit-tracer *only* to load transcoders and never imports
`ReplacementModel` / `AttributionGraph` / its intervention machinery. The scripts here
*do* import that machinery — purely to confirm our reimplementations match circuit-tracer
numerically. Nothing in `src/` depends on this folder.

## Scripts

- **`verify_attribution_graph.py`** — the attribution-graph faithfulness suite
  (11 checks, Qwen3-0.6B fp32 on CPU):
  1. *Forward exactness* — local replacement logits == real model logits.
  2. *Linearity identity* (exact, internal) — every feature pre-activation decomposes as
     `h_t = b_enc + <grad, emb> + Σ_l <grad_l, mlp_write_l>` through the frozen-linear
     backward; plus 200 emitted edge weights re-derived with plain per-target autograd.
  3. *Salient-logit parity* — same tokens/probs as circuit-tracer's `compute_salient_logits`.
  4. *Cross-check vs `circuit_tracer.attribute()`* (TransformerLens) — feature node sets,
     activations, edge weights on all shared pairs, influence ranking, and 0.8/0.98-pruned
     node sets.

  Status (2026-07-09, qwen3-0.6b, "The capital of France is"): **11/11 PASS** —
  node-set Jaccard 1.0000, edge cosine/pearson 1.0000 over 2.44M shared pairs, influence
  Spearman 1.0000, pruned sets identical (337 = 337).

- **`verify_intervention.py`** — runs our `run_feature_intervention`
  and circuit-tracer's `ReplacementModel.feature_intervention` on the same Qwen3 model,
  the same transcoders, the same prompt/features/values, with matching constrained layer
  ranges, then compares the steering effect on the logits. **Three cases** cover every
  intervention primitive the reproductions use: (A) single-feature negative steer
  (m=−2) constrained to its own layer; (B) a **two-layer coupled suppression stack**
  with the patch end layer above the top steer — exercising clean-anchored deltas for
  multiple features, the no-second-order-effects pin inside the range, and the
  within-range attention response (frozen patterns, live V); (C) a **`value=` donor
  injection** (a feature near-inactive at the target position set to an absolute
  positive value — the swap protocol's donor half) under the same constrained patching.
  Because the two sides use different model implementations (HF vs TransformerLens), it
  compares the **steering delta** (steered − clean), which cancels the baseline
  implementation offset, and PASSes when the deltas agree to within a few× the
  clean-logit noise floor (and cosine > 0.99). Exits 1 on any FAIL.

Run (fp32 — needs RAM for two 0.6B models + transcoders; on a Mac force CPU to avoid the
MPS allocator cap):

```bash
set -a; source .env; set +a
uv run python verification/verify_attribution_graph.py           # pins CPU itself
LLM_CIRCUITS_DEVICE=cpu uv run python verification/verify_intervention.py
```

## Known, documented divergences from the paper (matching circuit-tracer)

- **Error nodes can be pruned** — the paper says embedding *and error* nodes are never
  pruned; circuit-tracer only protects tokens/logits, and we match circuit-tracer.
- **Skip transcoders** — circuit-tracer keeps `W_skip` differentiable in the backward
  (part of J▼); our local replacement model detaches the whole reconstruction including
  the skip. Moot for the Qwen3 transcoder sets (`W_skip` absent), by construction
  identical results there; would matter if a skip-transcoder set is ever added.
- **Steering conventions** — ours is the additive-delta `m` (`a_new = (1+m)·a_clean`);
  the paper's multiplicative `M_paper = 1 + m_ours`; circuit-tracer takes an absolute
  `value`. See `llm_circuits.circuits.interventions`.
