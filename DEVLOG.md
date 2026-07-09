# DEVLOG

> **⏩ CURRENT STATE / HANDOFF — start here** (2026-07-09, end of the laptop session):
> see the last section, "Handoff: paper-exact reproduction suites → run on A100".
> Everything is committed on branch `tpu-iteration`; the suites are written and
> lint/test-clean but have NOT been executed yet — the next session (A100 GPU node)
> runs them and integrates results.

Development log for the verification pass over the re-implementation (task 4) and the
biology-paper reproductions (task 5), 2026-07-09. References: the methods paper
([methods.html](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)),
the biology paper ([biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)),
and circuit-tracer v0.3.1 (`safety-research/circuit-tracer @ e09b5f36`, the pinned dependency).

---

## Task 4 — Faithfulness of the re-implementation (attribution graphs + steering)

The whole algorithmic surface was compared line-by-line against the paper's formulas and
the installed circuit-tracer source: local replacement model construction, node types,
edge/adjacency computation, pruning, salient-logit selection, and feature interventions.

### Verified faithful (no change needed)

- **Local replacement model** (`circuits/local_replacement_model.py`): frozen attention
  patterns (pattern ⊗ live V/O), frozen RMSNorm denominators, detached reconstruction +
  detached error nodes with `error = captured_mlp_out − (decode + b_dec + skip)`, BOS
  splice zeroing position-0 errors — matches the paper's J▼ stop-gradient structure and
  circuit-tracer's "real forward, replaced backward" trick.
- **Edge computation** (`circuits/attribution_graph.py`): per-target gradient dotted with
  `a_s · W_dec` (features) / error vectors / embeddings; logit targets attributed as the
  **vocab-demeaned** logit; smallest logit set with cumprob ≥ 0.95 capped at 10.
- **Pruning** (`circuits/graph_pruning.py`): abs → row(target)-normalize (clamp 1e-10) →
  influence by iterated `w @ Â^k`; node threshold 0.8 / edge threshold 0.98 as cumulative
  cutoffs; edge score `Â_pruned · (influence + logit_prob)[target]`; iterative
  dangling-node cleanup — matches circuit-tracer exactly, **including its intentional
  deviation from the paper**: error nodes are *prunable* (the paper says never prune
  them); tokens/logits are protected.
- **Constrained interventions** (`circuits/interventions.py`): clean-anchored decoder
  deltas on the real model, attention-pattern freezing coupled to constrained ranges,
  LayerNorm frozen only when the range covers all layers — matches circuit-tracer's
  `feature_intervention` semantics. (Our `patch_end_layer` pins `[0, ℓ]`, equivalent to
  circuit-tracer's `[start, ℓ]` since nothing upstream of the first steered layer is
  perturbed.)

### Changes made

1. **`interventions.py` — corrected the steering-convention docs.** The module claimed our
   additive-delta `m` (`a_new = (1+m)·a_clean`) was "the paper's M convention". It is not:
   the paper's M is **multiplicative** (`a_new = M·a_clean`, so `M_paper = 1 + m_ours`;
   paper `−1×` flip == our `m=−2`), and circuit-tracer's API takes an absolute `value`.
   Off-by-one hazard when translating paper steering factors — now documented with a
   warning block and a conversion note on `FeatureIntervention`.
2. **`attribution_graph.py` — reject cross-layer transcoders loudly.** The Phase-4 source
   contribution used only the **offset-0 (self-layer) decoder vector**, silently dropping a
   CLT feature's writes to later layers (the paper's edge formula sums
   `a_s Σ_ℓ (W_dec^{ℓ_s→ℓ})ᵀ grad_ℓ` over *all* output layers). With the registry now
   Qwen3-only (per-layer transcoders) there is no CLT model to validate a fix against, so
   `build_attribution_graph` now raises `NotImplementedError` for CLTs instead of
   producing under-counted edges. Per-layer path is unaffected.
3. **`verification/verify_intervention.py` — repaired a stale API call.** The script passed
   `mode="steering-base-model"` to `run_feature_intervention`, a parameter that no longer
   exists (the call crashed with `TypeError`). Removed; re-run green (below).
4. **`verification/verify_attribution_graph.py` — NEW: attribution-graph faithfulness
   suite** (11 checks; imports circuit-tracer's `attribute`/`prune_graph`/
   `compute_salient_logits`, confined to `verification/` as per the repo rule):
   - forward exactness (local replacement logits == real logits);
   - an **exact linearity identity** validating the frozen-linear backward end-to-end
     (`h_t = b_enc + ⟨∇emb, emb⟩ + Σ_ℓ ⟨∇resid_ℓ, mlp_write_ℓ⟩`), independent of
     circuit-tracer;
   - 200 emitted edge weights re-derived with plain per-target autograd (validates the
     batched/vmapped edge machinery);
   - salient-logit parity;
   - full cross-check against `circuit_tracer.attribute()` (TransformerLens) on the same
     tokens/transcoders: node sets, activations, all shared edge weights, influence
     ranking, 0.8/0.98-pruned node sets.

### Verification results (Qwen3-0.6B, fp32, CPU, 2026-07-09)

`verify_attribution_graph.py` — **11/11 PASS**:

| Check | Result |
|---|---|
| local logits == original logits | max diff 1.2e-05 (logit scale 18.8) |
| linearity identity (24 targets) | worst rel err 1.7e-05 |
| 200 edge spot-checks vs plain autograd | worst rel err 3.7e-05 |
| salient logits | exact token + prob match |
| feature node sets (2546 vs 2546) | Jaccard 1.0000 |
| feature activations on common nodes | median rel diff 3.0e-06 |
| edge weights on 2,437,751 shared pairs | cosine 1.0000, pearson 1.0000, max dw 6.5e-04 |
| circuit-tracer edge mass missing from ours | 0.00 |
| influence ranking | Spearman 1.0000 |
| pruned node sets (337 vs 337) | Jaccard 1.0000 |

`verify_intervention.py` (steering, constrained range, m=−2) — **PASS**:
max |Δours − Δct| = 6e-05 over the whole vocab (== the HF-vs-TransformerLens noise
floor), cosine 1.00000, ‖delta‖ identical (37.104 both sides).

**Conclusion: the re-implementation is numerically equivalent to circuit-tracer** for
per-layer transcoders — same graphs, same pruning, same steering effects.

### Documented divergences (deliberate, now written down in `verification/README.md`)

- Error nodes prunable (matches circuit-tracer, deviates from the paper's prose).
- `W_skip` would be detached in our backward but kept differentiable in circuit-tracer's —
  moot for the Qwen3 transcoder sets (`W_skip` absent; confirmed at load).
- Steering conventions: ours `m` (delta) vs paper `M = 1+m` (multiplicative) vs
  circuit-tracer `value` (absolute).

---

## Task 5 — Addition + multilingual reproductions vs the biology paper

Checked `notebooks/addition.ipynb` and `notebooks/multilingual.ipynb` (both Qwen3-4b,
executed on the HPC) cell-by-cell against the paper's figures and claims, including a
visual comparison of every generated plot against the paper's operand plots / intervention
figures. **No contradictions with the paper were found** once model differences
(Qwen3-4b vs Claude 3.5 Haiku) are acknowledged — but the notebook *prose* was imprecise
or non-committal in places, and was corrected. **Markdown cells only were edited; no code
or outputs were changed, so all recorded results stand.**

### Addition (`addition.ipynb`)

Reproduction quality: **good, faithful adaptation**. Qwen3 tokenizes digit-by-digit, so
the paper's single-token parallel pathways appear as two per-digit circuits (magnitude
digit + teacher-forced ones digit) — an honest and well-documented adaptation.

- **Figures vs paper (visually compared):** the ones-digit panel reproduces the paper's
  operand-plot taxonomy — 5–6 clean **mod-10 anti-diagonal sum features** (paper's
  `sum = _5` class; ac10 ≈ 0.9, residue concentration up to 10.0), one genuine
  **lookup-table feature** (`L24 f163113`, a repeating grid of points jointly selective
  for each operand's residue — the analogue of the paper's `_6+_9`), and smeared
  low-precision magnitude diagonals in the magnitude panel (paper's `sum ~92` class).
- **Interventions:** m=−1 ablation of all 8 ones features flips the prediction 5→7 with
  p(5) 1.0→0.0 — the paper's causal-necessity claim reproduced.
- **Prose fixes (Summary cell):** the summary called the mod-10 *sum* features "lookup
  features, the paper's signature" — wrong taxonomy (the paper's lookup-table features
  are input-pair points; its sum features are the anti-diagonals). Corrected, and the
  actual lookup-table feature the notebook found is now called out. Also flagged
  `L34 f124769`'s auto-label `mod10-a(r9)` as a misclassification (visually a
  low-precision sum band, ac10 = 0.07), and listed what was *not* attempted (input-side
  lookup dominance, introspection dialogue, cross-context reuse à la Polymer citation) as
  scope gaps rather than contradictions.

### Multilingual (`multilingual.ipynb`)

Reproduction quality: **mixed — core structure reproduced, interventions partial, one
genuine (and interesting) model difference**. Outcome by experiment:

- **Graphs ✓** — shared multilingual say-big features across languages (`L30 f27666` in
  all three; `L31 f11436`, `L32 f100307` in two), language-specific final features.
- **Overlap-by-layer: partial.** A mid-network peak exists (layer 18/36, IoU 0.55) as the
  paper predicts, but the curve is *not* the paper's inverted-U — overlap is also high at
  layers 0–1 and 35, and `en-zh` is the **strongest** pair nearly everywhere (the paper's
  small model had EN-ZH *lowest*). Two causes noted in the notebook: no unrelated-prompt
  baseline control (the paper's control for shared surface tokens), and Qwen3's EN+ZH
  training mix. Baseline marked as TODO.
- **Operand swap (small→hot) ✓** — cleanest reproduction: top-1 becomes cold / f(roid) /
  冷 in all three languages at ×1 donor scale, exactly the paper's outcome.
- **Operation swap (antonym→synonym): partial.** The synonym direction appears (FR
  `Pet(it)`, ZH `小` at ×3) but never becomes top-1; ≥×6 collapses the distribution.
  Protocol difference recorded in the notebook: the paper *negatively steers* the source
  supernode at −5× (our `m=−6`) whereas the notebook only ablates (`m=−1`) — the milder
  suppression is a plausible cause; re-run with `m=−6` suggested.
- **Language swap: split.** EN→ZH works (大 top-1 @ 0.726 at ×3, operation/operand
  preserved — the paper's result); EN→FR fails (grand ≈ 0.09 max, degenerates at higher
  scales).
- **"English as the mechanistic default": does NOT hold on Qwen3-4b** — the notebook's
  own data shows mean direct effect zh 0.412 ≥ en 0.338 ≫ fr 0.143, with Chinese leading
  on the two strongest shared features. This is a legitimate **model difference**, not a
  paper contradiction (the paper's transferable claim is that *some* language acts as the
  default; for EN/ZH-trained Qwen3 that role is shared EN/ZH). The section was reframed
  from "English as the mechanistic default" to "Which language is the mechanistic
  default?" and now states the finding explicitly (with the small-sample caveat: only 4
  substantive shared features).

### Notebook edits (markdown only)

- `addition.ipynb`: cell 13 (Summary) rewritten — correct taxonomy, lookup-table feature
  called out, auto-label misclassification flagged, scope gaps listed.
- `multilingual.ipynb`: cell 6 (overlap expectations vs actual curve), cell 8
  (ablate-vs-negative-steer protocol note), cell 12 (default-language reframing + result),
  cell 14 (Summary rewritten with per-experiment outcomes).

### Follow-ups suggested (not done — need GPU time)

1. Re-run the multilingual operation swap with source suppression `m=−6` (paper's −5×).
2. Add the unrelated-prompt baseline to the overlap-by-layer experiment.
3. Addition: hunt input-side `_a+_b` lookup features in mid layers (upstream of the sum
   features) and try a Polymer-style cross-context reuse probe.

---

## Handoff: paper-exact reproduction suites → run on A100 (2026-07-09)

**Directive:** the reproduction must cover **all and exactly the paper's experiments**
(biology.html's Addition + Multilingual dives + the methods-supplement interventions
quoted there; the methods paper's *global virtual-weights* analysis is explicitly out of
scope). Protocol-deviating code was to be *modified*, not just reviewed. That code is now
written (commit `8d2cc0a`) but **has not been executed** — GPU was unavailable and the
TPU route was abandoned (below). The next session runs it on an **A100 node**.

### What is already done (this session)

1. **Faithfulness verified** (commit `a34df49`): our attribution graphs + steering are
   numerically equivalent to circuit-tracer (11/11 checks + steering PASS; details in the
   task-4 section above). Everything the suites compute rests on that verified stack.
2. **Paper-exact protocol layer + suites written** (commit `8d2cc0a`):
   - `examples/paper_multilingual_suite.py` — stages: `behavior, graphs, swap_operation,
     swap_operand, swap_language, overlap`. Paper protocols: source supernodes suppressed
     to **negative multiples** (antonym/language −5×, operand −0.5×; our `m = M_paper−1`),
     donors at +6×/+1.5×, strengths **swept** with crossover detection (paper ≈4×), all
     three language-swap directions, and the paper's overlap IOU (active-anywhere feature
     sets, translated paragraphs, **unrelated-pair baseline**) to be run at two scales.
   - `examples/paper_addition_suite.py` — stages: `accuracy, graphs, grids, interventions,
     steer_compare, polymer, intermediate, introspection, corpus`. Paper's raw
     `calc: a+b=` format (accuracy-gated chat fallback), lookup-point taxonomy class,
     input `_6`/`_9`/magnitude supernode suppressions **with downstream feature readout**,
     lookup-vs-sum −2× steering with smear metrics, Polymer reuse + suppression,
     `assert (4+5)*3 ==`, introspection dialogue, C4 dataset examples.
   - Support code: `notebooks/multilingual_helper.py` (paper_swap/paper_swap_sweep/
     early_language_detection_supernode/overlap_curves_paper + parallel paragraphs),
     `notebooks/addition_paper.py`, `notebooks/helper.py` (calc style, probe="last",
     lookup class), and core `run_feature_intervention(readout_layers=...)` which now
     fills `AblationResult.ablated_features` (the paper's "% of baseline" annotations).
3. **Reference material committed** to `notes/`: `methods_digest.md` (exact algorithms +
   circuit-tracer file:line cites) and `biology_digest.md` (every figure, quantitative
   result, intervention protocol and conclusion of the two dives, plus a
   model-specific-vs-generalizable checklist and figure URLs for visual comparison).
4. **TPU decision:** the gcp/ lifecycle itself is verified end-to-end (smoke-tested on a
   real v6e-1 this morning), but the suites were NOT run on TPU: Qwen3-4b's transcoder
   encoders (~30 GB bf16) + model exceed v6e-1's 32 GB HBM, and the XLA behaviour of the
   vmapped attribution backward is unresolved. Rather than engineering a lazy-encoder /
   CPU-hybrid path, the TPU was **torn down** (zero resources left) in favour of an A100.
   The TPU workflow remains available for forward-only workloads later.

### To continue on the A100 node

```bash
git pull                                  # branch: tpu-iteration @ 8d2cc0a or later
set -a; source .env; set +a               # HPC .env; do NOT set HF_HOME to an empty string
uv sync --all-groups                      # datasets was added to the notebook group

# Addition suite (~1-2 h; stages can be run separately / resumed — selections persist
# in artifacts/paper_addition/4b/state.json):
uv run python examples/paper_addition_suite.py --size 4b --stages all

# Multilingual suite on 4b:
uv run python examples/paper_multilingual_suite.py --size 4b --stages all

# Overlap at the second scale (the paper's scale comparison; cheap):
uv run python examples/paper_multilingual_suite.py --size 0.6b --stages overlap
```

Artifacts land under `artifacts/paper_{addition,multilingual}/<size>/` (JSON results,
PNG figures, explorer HTMLs). Runtime notes: each 8000-target graph build took ~65 s on
an A100 previously; the grids stage is 2×10,000 batched prompts; `corpus` streams C4
(network needed) — trim with `--corpus-docs` if slow.

### Then (the actual deliverable)

1. Compare every stage's output against the paper values in `notes/biology_digest.md`
   (§A.2/A.4 addition numbers + intervention table, §B.2/B.4 multilingual numbers +
   protocol table, §C for what is model-specific vs expected to transfer). Check the
   suite PNGs against the paper figures (URLs in the digest §D).
2. Update `notebooks/addition.ipynb` + `notebooks/multilingual.ipynb` to reflect the
   paper-exact results (either re-run them using the new helper protocols, or fold the
   suite artifacts in), and extend this DEVLOG with a per-experiment
   reproduced/partial/differs verdict table.
3. Known open questions the runs should answer: does the operation swap reach top-1 with
   the paper's −5×/+6× protocol (the m=−1 ablation didn't)? Does EN→FR language swap work
   with early-layer detection features? Does the overlap curve become paper-shaped once
   the baseline is subtracted (and does 4b > 0.6b hold, esp. for en-zh)? Does `calc:`
   accuracy hold up (else the chat fallback is used and must be reported)? Does the
   Polymer prompt complete "995"-style and reuse the calc lookup/sum features?

### State of the world

- Branch `tpu-iteration`, all work pushed. CI green (ruff + 86 tests) as of `8d2cc0a`.
- TPU: no resources exist; bucket `gs://dis-2026-zw499-tpu-store/llm-circuits/` holds
  only smoke-test artifacts. `.env` gotcha fixed this session: never set `HF_HOME=`
  (empty) — it roots a `hub/` HF cache inside the repo (now gitignored).
- Verification suites (`verification/`) run on CPU fp32 — see `verification/README.md`.
