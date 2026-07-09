# DEVLOG

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
