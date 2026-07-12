# DEVLOG

> **⏩ CURRENT STATE** (2026-07-11, after the FIFTH session, branch
> `repro/constrained-supernodes`): **reproduction v2 is live for ADDITION** —
> constrained patching only, supernodes loaded from a reviewed artifact
> (`notebooks/supernodes/addition_4b.json`; grid-evidenced full-graph scan, membership
> never by influence), graphs loaded from persisted dumps. v2 verdicts: input
> suppressions substitute crisply at the paper's strengths ([ℓ=16] `3`@0.96, [ℓ=9]
> `2`@0.94; ablation alone does nothing), the magnitude/ones dissociation holds at
> every strength with the two-band supernode ([ℓ=7], ones 0.999, low-prec → 42–81% at
> −2×), the **sum-side smear matches the paper** (width 5.26 at ℓ=35), polymer
> lookups-only is protocol-exact (`1`@0.83, all six sums 0%), and the **donor swap
> does not land at any ℓ** (p(`8`) ≤ 0.067) — with selection style/donor purity/
> protocol all controlled, that is now a clean model difference vs Haiku (v1's
> propagate route `8`@0.715 documents the mechanism). The supernode pipeline
> (build_supernode_inputs.py → build_supernodes.py → human review gate → notebooks
> load) is the durable workflow; MULTILINGUAL is parked mid-review (proposals emitted,
> approved:false; open items: detector off-graph fallback policy, the 0.95-prune
> regraph idea, per-language say-large flags). See "Fifth session" below.
>
> Previous state (2026-07-11, after the FOURTH A100 session, branch
> `fix/constrained-patching-pass`): the pre-report scan queue is closed. Every
> ✅-marked scan fix was independently re-verified against the code, the tests, the CI
> config and the committed notebook outputs (all hold), and the §3.1 decision is
> executed: **constrained patching is now the headline protocol** for every
> reproduction intervention — ℓ swept per the paper's recipe and stated next to every
> result, propagate kept as the no-pinning robustness variant (plumbing `bc71f40`,
> notebook code `cf9356b`, executed `5224c99`; 116 tests pass, zero cell errors, bf16
> continuity exact on a fifth A100). What the protocol switch changed: the
> magnitude/ones dissociation now holds at full paper strength for **both** supernodes
> (the loose set's propagate breakage was the unpinned cascade of the flipped always-on
> feature), the paper's **sum-side smear appears** (width 4.10 at ℓ=35), the polymer
> lookups-only indirect effect strengthens (`1` @ 0.83, sums 0%), and the language-swap
> null becomes **protocol-complete** (p_exp ≈ 0 at every ℓ × strength × direction ×
> format). What turns out to be **propagate-specific**: the 9+9-flavored `8`
> substitutions, the lookup swap's paper-margin landing (`8` @ 0.715 — constrained
> finds NO landing layer, p(`8`) ≤ 0.087), and the operation/operand swap crossovers
> (Qwen3's influence-top supernodes reach L27–35, leaving the paper's protocol
> little-to-no recompute room). Remaining open scan items are library/serve/
> housekeeping only (§3.2, §3.3, §5.x, §6). See "Fourth A100 session" below.
>
> Previous state (2026-07-10, after the THIRD A100 session, branch
> `handoff/a100-80gb-queue`): all five HANDOFF items are implemented, executed and
> written up. The strength ladder lives in `addition.ipynb` cell 11 (recheck numbers
> confirmed in-pipeline); `input_mag` is the paper-shaped two-band supernode
> (`band-a(~45)` + `band-b(~49)`) and the magnitude/ones dissociation **survives full
> paper strength** with it — the earlier m=−2 breakdown was selection pollution, and
> the magnitude verdict is upgraded to reproduced; the add-function hunt at the answer
> position is a measured negative; the computed-9 direct-weight screen finds no clean
> candidate ("partial" strengthened); the Fig B5-style readouts give the language-swap
> null its mechanism (the late zh say-stage never activates under full-strength
> early-detection drive — en→zh consolidates `large` 0.79→0.94). Both notebooks
> executed end-to-end on a fresh A100-80GB with zero cell errors; bf16 parity exact on
> every continuity number (lookup swap `8` @ 0.7148; polymer all-7 `1` @ 0.7467 and
> lookups-only `1` @ 0.7708 with sums → 0%; smears 1.26/1.03; operand crossovers
> 0.625×/1.0×/0.875×; scale 8b > 4b). **One item queued for the next session:** the
> constrained-patching pass — biology.html states all its interventions use
> "constrained patching" up to a chosen intervention layer, so the reproductions adopt
> it as the headline protocol (propagate kept as the no-pinning robustness variant);
> the executable spec lives in `DEVLOG_EXTRA.md` §0, row 3.1's Status cell. See
> "Third A100 session" below.
>
> Previous state (2026-07-10, after the VERIFICATION RECHECK session, branch
> `verify/paper-recheck`): both notebooks re-audited against the papers' own HTML +
> figure SVGs. Verdict prose corrected (biggest fix: the input suppressions had been
> compared at ablation strength — the paper's protocol is −1×/−2×, where the
> phenomenology partly breaks → "reproduced at reduced strength"); the paper's
> indirect-effect claim (Fig A5 panel 2) was measured paper-faithfully and
> **reproduces** (lookups-only → sum features 0%). All committed numbers verified
> (exact bf16 parity on a different A100). Its "HANDOFF — next A100-80GB session"
> section specced the five items resolved above.
>
> Previous state (2026-07-10, after the SECOND A100 session): both "Next-session
> queue" items are resolved — language-swap null survives the raw-format control
> (*differs* earned), and the 4b-vs-8b same-recipe run **reproduces the paper's scale
> claim**. Addition gaps closed; a non-idempotent duplicate-selection defect was found
> and fixed (four first-session numbers superseded — see the second-session section and
> the caution note on the first verdict table). Reproduction layout: two files per
> behavior under `notebooks/` (`addition.ipynb` + helper, `multilingual.ipynb` +
> helper); artifacts are per-node scratch under `artifacts/paper_*/<size>/` —
> regenerate by executing the notebooks.

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

---

## A100 session: suites executed + per-experiment verdicts (2026-07-09)

Environment: a **local A100-SXM4-80GB node** (no Slurm/TPU; fresh `.env` with all TPU
vars dropped — Qwen3 + mwhanna transcoder repos are public, so no `HF_TOKEN` needed).
`uv sync --all-groups`; 96 tests pass. Qwen3-4b bf16 (8 GB) + per-layer transcoders
(117 GB fp32 on disk → ~32 GB loaded) fit comfortably; each 8000-target graph build
takes ~95–125 s; a full 10,000-prompt grid probe ~2 min. Everything below is
Qwen3-4b + `mwhanna/qwen3-4b-transcoders` unless stated; paper = Claude 3.5 Haiku
(CLT), reference values in `notes/biology_digest.md`.

### Protocol/selection fixes made before the final runs (all four found by the first
### execution, not by review — each invalidated a paper experiment silently)

1. **Studied pair must be answered correctly** (suite premise, paper studies
   `36+59 → 95` which Haiku gets right). Qwen3-4b answers `calc: 36+59=` with `?\n\n`
   (the `_6+_9` class is only 46% correct in calc format; 77.7% overall), so the
   magnitude graph attributed a non-digit. `paper_addition_suite.py` now auto-switches
   to the **nearest correct pair with identical digit classes** (a%10, b%10, a+b all
   preserved → same `_6+_9 → sum=_95` structure): **46+49=95**. Persisted in
   `state.json` (`pair`, `pair_requested`).
2. **Input-feature operand grids probed at the wrong position.** With `style="calc"`
   the grids stage probed every group at the `=` token, where operand-token features
   are silent — `_6`/`_9` selections came out empty and both suppression experiments
   were skipped. Input groups are now probed at their **peak over prompt positions**
   (`grids_peak`); answer/midlayer groups keep the functional-position probes.
3. **Input supernodes must be selected from EARLY layers.** Influence-top-12 at the
   operand digit positions is dominated by late-layer aggregation features (L24–L35),
   which crowd out the paper's detokenization-level `_6`/`_9`/`~magnitude` features
   (L0–L7); `_9` was empty and `inhibit_magnitude` was polluted (it flipped the ones
   digit — an artifact). `position_features` gained a `max_layer` cap; the suite uses
   `n_layers//4` with a 16-deep pool.
4. **Language-detection supernodes were EMPTY** — `early_language_detection_supernode`
   filtered a *global* activation-top-40 (all late layers) down to early layers,
   leaving zero features per language, so the first language-swap run intervened on
   nothing (baselines unmoved, p_expected = 0). `position_supernode` gained a
   `max_layer` argument restricting the candidate pool itself; rerun below.

Two paper experiments needed new scripts (the digit-split adaptation makes them
position-sensitive):

- **`examples/paper_addition_polymer_probe.py`** — the polymer reuse test at the
  citation's **ones moment**. Haiku predicts `995` as one token; Qwen3 emits digits
  one at a time, so the `_6+_9 → _5` lookup can only fire when predicting the final
  `5`, i.e. on `"…Polymer, 36, 837, 199"`. (The suite's pruned-graph-membership test
  at the bare prompt — predicting the first `9` — is the wrong moment and finds
  nothing; the probe also confirms that negative directly.)
- **`examples/paper_addition_swap_addendum.py`** — the Fig A5 lookup swap needs a
  verified `_9+_9` donor. The 46+49 selection has none (its auto-labelled (9,9)
  candidate is really an ends-in-0/5 sum feature — operand-grid inspection, not
  labels, is the arbiter). The addendum builds the `calc: 49+49=` ones graph,
  grid-verifies its mid-layer candidates (3 clean `lookup(9,9)` point-lattices:
  L25f42677, L25f127159, L23f25848), and runs the swap with those donors.

### Addition verdicts (paper §A vs Qwen3-4b, studied pair 46+49=95)

> ⚠️ **Four rows below are superseded** — this session's selections carried a
> non-idempotent duplicate (features steered at double strength) and the lookup-swap
> donor filter was loose. Corrected numbers (second A100 session): lookup swap
> **`8` @ 0.715** (not 0.845); lookup neg-steering **flips to `1` @ 0.88, width 1.26**
> (not "smears to width 2.0"); polymer suppression `1` @ 0.75 on 7 distinct features;
> magnitude-inhibition readout re-measured with the paper's low-precision readout.
> The rest of the table stands (re-executed identically).
>
> ⚠️ **Strength annotation (recheck 2026-07-10):** the suppress-`_6`/`_9`/magnitude
> rows below quote **m=−1 (ablation)** results — weaker than the paper's protocol (its
> prose says "negative of its original value", −1× ⇒ m=−2; its figure
> `patching-arithmetic-svg` is annotated −2× ⇒ m=−3). At m=−2 the phenomenology
> differs (suppress-`_6` → `2` @ 0.99, not `8`; magnitude inhibition flips the ones
> digit). Verdicts re-framed as "reproduced at reduced strength" — see the recheck
> section. (Also: the `8 @ 0.36` here was the retired suite's number; the committed
> notebook reads `8 @ 0.29`.)

| Experiment | Paper (Haiku) | Qwen3-4b (this session) | Verdict |
|---|---|---|---|
| `calc:` accuracy (10,000 prompts) | implied high | 77.7% overall — format usable, no chat fallback; but 36+59 itself wrong (`?\n\n`), `_6+_9` class 46% → pair switched to 46+49 | **partial** (format holds; weaker arithmetic) |
| Two-pathway graph (Fig A2) | input → add-function → lookup → sum, one token | two per-digit circuits (digit-split adaptation): first-digit graph answers `9`, teacher-forced ones graph answers `5`; 1167-feature pruned ones graph | **reproduced** (adapted) |
| Operand-plot taxonomy (Fig A1) | diagonals=sum, points=lookup, repeating=mod-10, smears=low-precision, stripes=operand | all classes found: 4 clean `sum=_5` anti-diagonals + `sum=_95`-like single diagonals (answer panel); 5 lookup point-lattices incl. (9,9), (5,9), (6,6), (3,3) + a clean `a+b≈95` band (midlayer); `_6` mod-10 lattices, `~46` magnitude bands, exact-46/49 crosses (input panel) | **reproduced** |
| Suppress `_6` input | output 98 (ones = 9+9→8) | m=−1: ones digit → **`8`** @ 0.36 (exactly the 9+9 phenomenology); ones-path readout: lookups 21–46%, two sum feats 0% | **reproduced** |
| Suppress `_9` input | output 91 ("not 92 — grain of salt") | m=−1: `5` stays top but 0.999→0.52 (sum=_5 feats → 0%; redundancy resists); m=−2: → `3` @ 0.96. Like the paper, does NOT follow 6+6-numerology | **reproduced** (same caveat) |
| Inhibit `~30`/`~59` magnitude | low-precision path suppressed, **ones path intact** | early-layer `~46`-band supernode, m=−1: ones digit **unchanged** (`5` @ 0.995), all ones-path readouts 101–133% | **reproduced** |
| Neg-steer lookup vs sum (−2× = m=−3) | lookup smears result over ~5; sum smears wider | lookup: digit width 1.0→2.0 (top `7` 0.67 / `3` 0.19); sum: width ~1.03, flips sharply to `3` @ 0.99 | **partial/differs** (lookup smears less; sum flips instead of smearing) |
| Polymer completion (Fig A4) | `995` @ 98.6% | `995.` greedy; ones moment `5` @ 0.984 | **reproduced** |
| Polymer reuse of calc features | same `_6+_9` lookup active in citation graph | at the **ones moment**: 8/10 calc lookup/sum features active (`sum=_5` L34f51125 @ 74.5, lookup L24f163113 @ 12.8, …); at the bare-prompt magnitude moment: none | **reproduced** (needs the per-digit position) |
| Polymer −2× suppression (Fig A5) | 995 → 997 top @ 54.8%, sum/say-995 → 0% | m=−3 on the 8 active: `5` @ 0.984 → **`1`** @ 0.56 (correct year destroyed) | **reproduced** (analog) |
| Polymer lookup swap `_6+_9`→`_9+_9` (Fig A5) | 995 → **998** @ 66.6% | flip 3 active (6,9) lookups + inject 3 verified (9,9) donors at 1×: `5` @ 0.982 → **`8`** @ 0.845 (= …1998) | **reproduced** |
| Intermediate `assert (4+5)*3==` (Fig A6) | `27`; computed-9 intermediate features | completes ` 27`; label-based hunt finds only 2–3 digit-logit features at influence ≤ 0.003 — no compelling computed-9 story (paper hedged its own suppression mechanism here) | **partial** (behavior ✓, feature story weak) |
| Introspection dialogue | narrates the carry algorithm it doesn't use | answers `95`, then *"I added 46 and 49 together: 46 + 49 = 95."* — a non-explanation; no carry narration | **partial** (mismatch motif trivially holds — no metacognitive insight; the specific carry-narration behavior is absent) |
| Corpus examples (Fig A3) | lookup feature fires on citations/tables/dates | C4 scan: lookup L24f163113 on dates/totals ("April 25, 2⟦0⟧10", "1⟦,⟧016,650"), L25f145046 on journal page ranges ("pp. 43⟦-⟧58") and prices, L27f77434 on complementary percentages ("46.4% … ⟦ ⟧53.6%") | **reproduced** |

### Multilingual verdicts (paper §B vs Qwen3-4b)

| Experiment | Paper (Haiku) | Qwen3-4b (this session) | Verdict |
|---|---|---|---|
| Behavior | big/grand/大 | antonym large/Grand/大 ✓; hot-antonym cold/F(roid)/冷 ✓; synonym: EN `tiny` ✓ but FR/ZH **echo the operand** (petit→petit, 小→小) — degenerate synonym mode | **reproduced** (antonym); synonym degenerate in FR/ZH |
| Shared supernodes (20/27, 10/27 in pruned graphs) | multilingual antonym/say-large core | **107 features in all three pruned graphs** (437–621 per graph); say-big trio in all three: L30f27666 (巨大/giant), L31f11436 (large/big/大), L32f100307 (bigger/太大/大); task-circuit pairwise: en∩zh 244 > en∩fr 139 > fr∩zh 114 | **reproduced** (structure) |
| Operation swap −5×/+6×, crossover ≈4× | little/min./微 top-1, upstream intact | at moderate strengths the answer flips to the model's **own synonym-mode output**: FR `pet`+`Pet` 0.65 top-1 @ 1×, EN `small` (echo) 0.98 @ 2×, ZH 小 0.358 (2nd) @ 1.5×; technical crossovers 1–3× (paper ≈4×); at the paper's full ±5–6× all languages **degenerate to junk** (Haiku stayed coherent) | **partial** (operation independently editable ✓; full paper strength over-drives the 4B model) |
| Operand swap −0.5×/+1.5× (Fig B4) | cold / f[roid] / 冷 | EN `cold` @ 0.924 (crossover 0.625×), FR **`f`** @ 0.986 (1.0×; the paper's own figure shows "f[roid]"), ZH `冷` @ 0.998 (0.875×) | **reproduced** (all three) |
| Language swap −5×/+6× (Fig B5) | EN→ZH 大, FR→EN big, ZH→FR grand; operation+operand preserved | with **populated** early-layer detection supernodes (12/lang, L6–11 — after fix #4): **no effect in any direction** — baselines stay top-1 (large 0.948 / Grand 0.996 / 大 1.0), p_expected = 0.000 at every strength. The old notebook's any-layer language-specific final features DID move EN→ZH (大 top-1 @ 0.726 at ×3; EN→FR failed) → on Qwen3-4b output language is set by **late say-X-in-language features**, not early detection features. Format caveat: our chat-template final token is not the paper's content-bearing open quote | **differs** (causal handle sits later than in Haiku) |
| Overlap-by-layer IOU + baseline (Fig B7) | mid-layer peak (0.2–0.3), baselines much lower, ends low | 4b: mid-layer peak ✓ (mean 0.348 @ L22 vs baseline 0.277); baseline-subtracted curve is paper-shaped (mid ≈ 0.10 vs ends ≈ 0.02) and ordering **en-fr > en-zh > fr-zh = the paper's** (the old top-k method's en-zh anomaly disappears under the paper protocol) | **reproduced** (4b) |
| Scale claim (Haiku ≫ 18L, esp. EN-ZH) | overlap grows with scale | **inverted on our pair**: 0.6b ≥ 4b (baseline-subtracted mid-mean 0.105 vs 0.091; raw peaks 0.48–0.52 vs 0.33–0.39). Caveat: 0.6b uses the `-lowl0` transcoder recipe (different sparsity/dictionary) — granularity confounds IOU, so treat as **inconclusive**, not a contradiction | **not reproduced** (confounded) |

### Answers to the handoff's open questions

1. *Operation swap top-1 with −5×/+6×?* Yes at moderate strengths for FR (petit @ 1×)
   and effectively EN (`small` echo @ 2×); ZH reaches 2nd place — but the swap lands on
   Qwen3-4b's own (echo) synonym behavior, and the full paper strengths destroy the
   distribution. The m=−1-only ablation of the old notebook was indeed the weaker
   protocol.
2. *EN→FR language swap with early detection features?* No — and with the paper's
   early-layer supernodes populated, **no direction works at all** (verdict row above).
   The language swap is the one paper intervention that does not transfer to Qwen3-4b
   chat prompts; the causal handle is late say-X features (any-layer selection moved
   EN→ZH previously), not early language detection.
3. *Overlap paper-shaped after baseline?* Yes — and the pair ordering becomes the
   paper's (en-fr strongest). 4b > 0.6b does **not** hold (inverted; dictionary
   confound noted).
4. *`calc:` accuracy?* 77.7% → format kept, no chat fallback; but the paper's exact
   pair 36+59 is wrong (46% class accuracy) → studied pair auto-switched to 46+49.
5. *Polymer "995" + reuse?* Completion yes; reuse **yes at the ones moment** (8/10
   features, causal at m=−3), invisible at the bare-prompt magnitude moment.

### Not attempted / instrumentation gaps (scope, not contradictions)

- The paper's magnitude-inhibition readout covers the *low-precision* features
  themselves; our readout list covers the ones-path lookup/sum features only (the
  low-precision side is evidenced by the unchanged ones digit + the first-digit
  circuit's existence, not by a % table).
- "Add-function" features (operand-stripe class at the `+`/`=` positions) were not
  probed as their own group.
- The introspection prompt was asked once (greedy); no sampling over phrasings.

### Next-session queue (both items resolved — see the second-A100-session section below)

- **Language swap: rerun on RAW open-quote prompts before trusting the "differs"
  verdict.** The paper's recipient position is a content-bearing open-quote token
  (`…the opposite of "small" is "`), where the open-quote-in-language-X detection
  features live; our chat-template prompts end on assistant-header tokens that carry no
  quote at all — so the early-layer supernodes may have had nothing to grab. Control:
  build the language-detection supernodes and run the −5×/+6× swap on raw
  `The opposite of "small" is "` prompts (accepting weaker task behavior). If it still
  does nothing, "the causal handle sits later in Qwen3" is earned; until then the
  verdict is *unresolved-format-confound*, not *differs*.
- **Scale claim: de-confound with a same-recipe pair — rerun the overlap IOU at 4b vs
  8b.** The "not reproduced (confounded)" verdict compared 0.6b (`-lowl0` transcoder
  recipe) against 4b (plain recipe), so dictionary granularity/sparsity confounds the
  IOU. The registry has a same-recipe pair: `mwhanna/qwen3-4b-transcoders` vs
  `mwhanna/qwen3-8b-transcoders` (both plain). Swap the 0.6b for 8b in
  `multilingual.ipynb`'s scale-comparison section (it already frees the 4b before
  loading the comparison model; overlap is forward-only, so an A100-80GB fits 8b bf16 +
  lazy-decoder transcoders). If 8b > 4b on the baseline-subtracted mid-layer IOU
  (especially en-zh / fr-zh), the paper's scale claim reproduces; if the inversion
  persists on a same-recipe pair, it is a real Qwen3-vs-Claude difference worth
  reporting.

---

## Repo cleanup: reproduction consolidated into notebooks/ (2026-07-09, same session)

The reproduction had spread across `examples/` (four `paper_*.py` stage-scripts, a
TPU/batch-era artifact) and `notebooks/` (two notebooks + three helper modules). It now
lives **only under `notebooks/`, two files per behavior**:

- `addition.ipynb` + `addition_helper.py` — the helper merges the former `helper.py`,
  `addition_paper.py`, and the non-CLI logic of `examples/paper_addition_suite.py`,
  `paper_addition_polymer_probe.py`, `paper_addition_swap_addendum.py` (all six retired
  files deleted). The notebook runs every stage of the paper dive inline and was
  re-executed end-to-end on the A100.
- `multilingual.ipynb` + `multilingual_helper.py` — the helper gained the sweep
  orchestration (`run_swap_sweeps`) from the retired `examples/paper_multilingual_suite.py`
  and a `size_key` parameter for `build_graph`; the notebook runs behavior → graphs +
  shared-core analysis → the three paper swaps → overlap (+ baseline) → default-language →
  the 0.6b scale comparison (frees the 4b and loads 0.6b in-notebook), re-executed
  end-to-end.

`examples/` keeps only generic machinery demos (`compare_*`, `graph_explorer`,
`layer_range_sweep`, `progressive_steering`, `steering_explorer`, `tpu_smoke_test`, and
`addition_circuit.py`, which those demos consume artifacts from — it demos the toolkit
pipeline, not the paper protocol). Updated pointers: `CLAUDE.md` (architecture bullets),
`README.md`, `notebooks/README.md`, both notebook sbatch wrappers (artifact paths are now
`artifacts/paper_{addition,multilingual}/<size>/`). The historical sections above refer to
the retired `examples/paper_*.py` paths; their logic lives on in the two helpers.

---

## Second A100 session: queue items resolved (language-swap control, 4b-vs-8b scale) + addition gaps closed (2026-07-09/10)

Environment: a **fresh A100-80GB node** (Lightning studio, no Slurm) — no `.env`, no
`.venv`, no HF cache, and the previous node's `artifacts/` gone. Rebuilt from scratch:
`uv sync --all-groups` (ruff clean, 95 passed / 1 skipped), minimal `.env`, ~180 GB of
weights re-downloaded. Branch `feat/next-session`.

**Infra findings (durable):**

- The mwhanna transcoder repos store **bf16** weights; circuit-tracer's cache converts
  to fp32 by default — a pure upcast that doubles disk (8b: 97 → 193 GB) and would not
  have fit alongside the 4b set. `cache_transcoder()` now takes `dtype=`; both sets are
  cached bf16 (60 + 97 GB), which also halves lazy-decoder disk reads (multilingual
  notebook wall time dropped ~55 → ~48 min for the old sections; ~60 min with the two
  new ones).
- circuit-tracer's `save_transcoders_to_cache` takes the snapshot path for these repos
  (no `transcoders:` key in config.yaml) and therefore **never deletes the hub-side
  copy** — delete `~/.cache/huggingface/hub/models--mwhanna--*` manually after caching
  or pay 2× disk. Feature labels are unaffected (separate `.cache/feature_labels`).
- Both dictionaries are **163,840 features/layer** (4b and 8b, per safetensors headers)
  and both models are 36 layers — the same-recipe scale pair is granularity-matched by
  construction.
- Determinism: the re-executed chat-format sections reproduced the previous session
  **exactly** (identical pruned node sets 621/437/544 with 107 shared, identical
  crossovers, identical overlap curves and direct-effect values) — no bf16 drift.
- TODO.md removed (stale); its eager-decoder lesson moved to CLAUDE.md "Performance
  notes".

### Queue item 1 — language swap rerun on RAW open-quote prompts: null SURVIVES; "differs" is earned

Raw behavior first (new `tokenize_raw` + `raw=` path in `multilingual_helper`, sink-token
prepend, `n_bos_tokens=1`): the paper's exact prompts work — antonym **large / grand / 大
@ 0.77 / 0.99 / 0.99** (single tokens; no task-quality penalty for dropping the chat
template). The FR synonym echo persists raw (`pet` 0.86 — a model behavior, not a chat
artifact); ZH echo is borderline (小 0.57 vs 微 0.31).

The control (multilingual.ipynb §G): language-unique early supernodes built on the raw
prompts are **well populated** (en 9 / fr 12 / zh 12 features, L4–L11, acts 2–4.6) and sit
on a content-bearing final `"` token — exactly the paper's recipient. The −5×/+6× sweep
still does **nothing in any direction**: p(target-language answer) = 0.000 at every
strength. en→zh *strengthens* `large` (0.77→0.94); fr→en leaves `grand` at 0.97; zh→fr
at ≥4.5× degrades toward literal quote tokens (`"` 0.40) — breaking the "inside a quote"
representation rather than switching language. With the format confound eliminated, the
verdict is now an earned **differs**: on Qwen3-4b, output language is causally carried by
late say-X-in-language features (the any-layer variant moved EN→ZH previously), not the
paper's early detection features.

Bonus from the same infrastructure — **operation swap in the paper's raw format** (its
synonym donor prompt is raw EN `A synonym of "small" is "`): reaches **top-1 in all three
languages** at 1–2× (EN `small` 0.885 @ 2×, FR `pet` 0.730 @ 1×, ZH 小 0.911 @ 1.5×),
landing on the model's own echo-mode synonym; the paper's full ±5–6× still over-drives
into junk in both formats. Verdict upgraded partial → **reproduced** (adapted), with the
robustness gap retained as a genuine 4B-vs-Haiku capacity difference.

### Queue item 2 — scale claim de-confounded: 8b > 4b, paper direction reproduced

multilingual.ipynb §H swaps the 0.6b comparison for the same-recipe
`qwen3-8b-transcoders` (lazy decoders; encode-only overlap fits comfortably).
Baseline-subtracted mid-third IOU: **8b > 4b on every pair** — en-fr 0.132 vs 0.107,
en-zh 0.098 vs 0.089, fr-zh 0.082 vs 0.077 (mean 0.104 vs 0.091). The previous
"0.6b ≥ 4b" inversion was therefore the `-lowl0` recipe confound, and the paper's
overlap-grows-with-scale claim **reproduces on a clean pair**. Caveats recorded: the
gain concentrates on en-fr (+23%) rather than the paper's emphasized non-alphabet pairs
(en-zh +10%, fr-zh +6%), and 8b activates more features per layer (median 1206 vs 935;
`overlap_curves_paper` now logs `set-size`) — the mechanical IOU component of that is
netted out by the paper's unrelated-pair baseline.

### Addition: three instrumentation gaps closed + a consolidation bug caught by the recheck

addition.ipynb re-executed end-to-end three times (~40 min each; the second run after
the donor fix below, the third after the selection dedup). Stability: studied pair
re-selects **46+49** (77.7% / 46.0% accuracies identical); the first re-run reproduced
the previous session's committed outputs essentially exactly — including, it turned
out, two of its protocol defects.

1. **Magnitude-inhibition low-precision readout** (the paper's actual readout, previously
   missing): the same `input_mag` suppression run on the *first-digit* graph with the
   magnitude-class answer features as readout. At ablation level (m=−1 — ⚠️ recheck
   2026-07-10: this is *below* the paper's strength, not "the paper's level" as this
   entry originally said; the methods prose is −1× ⇒ m=−2 and its figure −2× ⇒ m=−3,
   see the recheck section), with the
   selection dedup below in place, the dissociation is clean on both halves: the three
   magnitude-band features drop to **54–66%** and the first digit weakens `9` 0.99→0.40,
   while the ones path stays 85–116% intact (`5` @ 0.998) — the paper's claim at matched
   strength.
2. **Add-function group** (paper's remaining input class, previously unprobed): 16
   influence-top features at the `+`/`=` operator tokens; **15/16 grid as operand-
   uniform** (a/b concentration ≈ 1.00–1.17 — fire regardless of operands), mostly
   L0–L5. The paper's "this is addition" class exists on Qwen3 with the expected
   signature.
3. **Introspection sampling** (6 phrasings × 2 seeds, T=0.7; answer turn stays greedy):
   the greedy-only conclusion was **wrong** — about half the sampled explanations
   narrate the schoolbook place-value algorithm ("add the tens: 40+40=80, add the ones:
   6+9=15, …"), which the circuits (parallel lookup/sum features) do not implement.
   That is the paper's mismatch motif, hidden by greedy decoding. Verdict upgraded
   partial → reproduced.

**Lookup-swap discrepancy found by the verdict recheck** (donor filter fixed,
`e8cf4a9`): the consolidated Fig A5 cell tags any active `lookup`-class candidate as a
donor — its own comment says "verified `_9+_9`" but the (9,9) grid check was never in
the filter (the retired addendum's filter was equally loose; its candidate pool just
happened to contain only (9,9) lookups). The notebook had therefore been injecting
(8,7)/(4,4)/(2,2) lookups alongside the three verified (9,9) features since the
consolidation — including in the previous session's committed outputs (`8` @ 0.39,
flattered by the (4,4)→8 donor) — while this DEVLOG's table quoted the addendum's 0.845
(which also rested on the retired suite's `state.json` selections; not recoverable in
the notebook pipeline). With donor purity restored the swap first read `8` tie-top-1 @
0.237 — still short of the paper. The remaining culprit was what the recheck initially
filed as cosmetic: `sel["lookup"]` (and `sel["input_mag"]`) carried a **duplicate** —
a feature can classify into the same class from two grid panels — and duplicates are
**not idempotent**: real-model steering applies *additive* decoder deltas, so a doubled
entry doubles the flip (effective m=−4/−6 on L24f163113). Deduping the selections (one
line in cell 8) moved four measurements, all toward the paper: the lookup swap becomes
**`8` @ 0.715 top-1 — the paper's 66.6% margin matched** (verdict back to
**reproduced**); lookup neg-steering reads `1` @ 0.88 at width 1.26 (the earlier "width
2.0 smear onto 7" was the double-flip artifact — the smear row is now a clean
"differs: flips rather than smears, both classes"); the polymer suppression is `1` @
0.75 on the 7 distinct active features; and the magnitude readouts are the gap-1
numbers above. The duplicate predates this session (the previous session's committed
outputs carry it), so those four numbers in the first-A100-session table each reflect
one feature steered at double strength.

### State of the world (after the second A100 session)

- Branch `feat/next-session` (session commits `3441e31..ceed332`); ruff clean, 95
  tests pass; both notebooks executed end-to-end on this node with zero cell errors.
  Artifacts regenerated under `artifacts/paper_{addition,multilingual}/{4b,8b}/`.
- Verdict tables live in each notebook's Summary (updated in place); this section
  resolves both items of the "Next-session queue" above.

---

## Verification recheck: reproductions re-audited from the paper sources (2026-07-10)

Branch `verify/paper-recheck`, fresh **A100-40GB** node. Independent audit of both
notebooks + helpers against re-downloaded copies of biology.html, methods.html, and
every figure SVG (Figma exports carry the intervention multiples and outcome
percentages as vector text) — deliberately not trusting `notes/biology_digest.md` or
the notebooks' own protocol notes. Every generated figure was compared against the
paper's; every helper on the causal path re-read against the paper's protocol. Full
report: **`notes/verification_recheck.md`**. Corrections were applied as
**markdown-only** notebook edits (code/outputs untouched, the task-5 practice); the
missing paper-strength measurements were re-run standalone (below).

### What survives scrutiny (checked against the sources, not just re-read)

The digest is accurate on every checked number; multilingual protocols are faithful
end-to-end (endpoints, m = M_paper−1 conversions, sweeps/crossovers, raw-format
control, same-recipe scale pair) — the operand swap, overlap(+baseline), scale
direction, and the earned language-swap **differs** all stand. The paper's own Fig B3
ZH panel shows the operand echo (小) rising under the operation swap — Qwen3's
echo-synonym landing has a Haiku analogue (noted in the Summary). Addition: graphs,
taxonomy geometry (panel-by-panel), polymer completion/reuse, the protocol-exact
lookup swap, introspection, corpus reuse all stand.

### Corrections (details + diffs in the report; notebook Summaries updated in place)

1. **Input-suppression strength was mislabeled** (the biggest find). The methods prose
   suppresses supernodes to the *negative of the original value* (−1× ⇒ m=−2); its
   own figure `patching-arithmetic-svg` says **−2×** (⇒ m=−3); ablation (m=−1) is
   weaker than either — yet the Summary quoted the m=−1 numbers as the reproduction
   and this DEVLOG had called m=−1 "the paper's level" (inline-corrected above).
   Verdicts re-framed **reproduced at reduced strength**.
2. **The polymer "−2× suppression" cell steers lookups+sums together** — not Fig A5
   panel 2 (lookups only, sum features as readout). Resolved by the recheck run below:
   the paper's indirect-effect architecture **reproduces**.
3. **"Add-function class" was a taxonomy mislabel** — operand-uniform operator-token
   features are the paper's *Mostly Active* signature; the stripe-like Add Function
   class (`add _9`) has not been hunted and is now recorded as a scope gap.
4. **`input_mag` supernode pollution**: the `mixed` catch-all includes an always-on
   flat feature (`L0f116505` — the very feature the duplicate-dedup caught) and
   ripple features; less surgical than the paper's clean `~30`/`~59` bands.
   Band-criterion re-selection queued.
5. **"(6,9) lookups" wording**: the suppressed lookup features auto-label
   (3,9)/(5,9)/(6,6) — Qwen3's lookups are coarser **multi-pair lattices** whose
   receptive fields include (6,9); Haiku's read as single-pair. Wording fixed; the
   model nuance is worth keeping.
6. **Intermediate-computation hunt bias**: scanning label *top* logits cannot find
   the paper's computed-9 feature, whose signature is a *negative* "9" output weight.
   Caveat added (verdict stays partial).
7. Doc staleness: first-session table's `8 @ 0.36` was the retired suite's number
   (notebook: 0.285); its default-language means were pre-consolidation (notebook:
   zh 0.766 / en 0.664 / fr 0.309, same ordering). Digest gained the −2× figure
   annotation + the two methods-figure slugs.
8. Multilingual Summary now also notes: crossovers are early and language-dependent
   (1–3×) vs the paper's "fairly consistent ≈4×".

### Recheck measurements (A100-40GB; selections reconstructed from committed outputs)

Selections rebuilt feature-by-feature from the committed notebook (cell-8 sizes,
cell-11 readout keys, cell-9 panel labels); m-mode steering needs no stored
activations. **Parity is exact**: every number that overlaps the committed run
reproduces to the printed precision (e.g. suppress-`_6` m=−1 `8` @ 0.285; magnitude
m=−2 `3` @ 0.5368; polymer all-7 `1` @ 0.7467; calc lookup steer `1` @ 0.8841,
width 1.26) — bf16 determinism holds across a different A100 variant.

Input suppressions across the strength ladder (ones-digit answer; baseline `5` @ 0.999):

| Experiment | m=−1 (ablation; the old comparison) | m=−2 (paper prose −1×) | m=−3 (paper figure −2×) |
|---|---|---|---|
| suppress `_6` (paper: 98) | **`8`** @ 0.29 — the 9+9 story | `2` @ 0.99 | `2` @ 0.99 |
| suppress `_9` (paper: 91) | `5` @ 0.52 | `3` @ 0.96 | junk (`,` 0.39, `""` 0.35) |
| inhibit magnitude (paper: ones intact) | `5` @ 0.998, ones-path 85–116% — clean dissociation | `3` @ 0.54, sums 0–23% — broken | junk (flat, top `""` 0.08) |
| … first-digit readout | `9` 0.99→0.40, mag features 54–66% | `1` @ 0.21, mag 0% | flat (`1` @ 0.09) |

> ⚠️ 2026-07-10, pre-report scan (`DEVLOG_EXTRA.md` §1.1/§1.2): the suppress-`_9` row
> here (and everywhere before the third session's addendum) was steered at the
> **teacher-forced answer position**, not the operand's ones digit — a
> `digit_token_positions` defect, since fixed; the re-measured operand-position ladder
> is in the third-session addendum. And the first-digit readout row's `0.99` baseline
> is the *ones*-prompt value; the first-digit graph's own baseline is `9` @ **0.5626**
> (steered values in the row are correct).

Reading: the paper's qualitative phenomenology lives at **ablation** strength on
Qwen3-4B and progressively shatters at the paper's −1×/−2× — the same
over-drive-at-paper-strength pattern as the multilingual grafts. This is now the
stated verdict framing ("reproduced at reduced strength").

Polymer + calc, the paper-faithful lookups-only runs (m=−3 = figure −2×):

| Run | Output | Sum-feature readout |
|---|---|---|
| polymer ones moment, suppress the **3 active lookups only** | `5` @ 0.982 → **`1`** @ 0.77 (199⟦1⟧ — magnitude-consistent nearby year, cf. the paper's 997 @ 54.8%) | **all 4 active sum features → 0%** — the paper's number |
| (parity: committed all-7 variant) | `1` @ 0.7467 ✓ | — |
| calc ones prompt, suppress the 5 lookup-class features only | `1` @ 0.88, width 1.26 (the committed smear row) | **all 4 sum features → 0%** |

Reading: Fig A5 panel 2's **weak-direct / strong-indirect** claim reproduces —
suppressing only the lookup stage silences the sum stage completely and changes the
answer; the sums add nothing beyond what the lookups already carry (0.771 vs 0.747).
Polymer-suppression verdict upgraded from "analog" to **reproduced**; the smear-row
"differs" gains a sharper statement (the flip to `1` happens with the sum stage fully
silenced).

### HANDOFF — next A100-80GB session: five queued items (specs, no code written yet)

The session's job: implement the five items below (small, localized edits — the heavy
machinery all exists), execute **both notebooks end-to-end**, eyeball the new panels,
refresh the Summary verdict rows and extend this DEVLOG. Reference numbers to reproduce
are in the two recheck tables above (bf16 parity across A100 variants was exact, so the
notebook's numbers should match them to the printed digit).

**Bootstrap on a fresh node** (~30–40 min, mostly downloads): `uv sync --all-groups`;
minimal `.env` (never set an empty `HF_HOME`); pre-cache both transcoder sets at bf16 —
`cache_transcoder("mwhanna/qwen3-4b-transcoders", dtype=torch.bfloat16)` and the same
for `qwen3-8b` (multilingual §H needs it) — then delete
`~/.cache/huggingface/hub/models--mwhanna--*` or pay 2× disk (60+97 GB cached).
Execution: `addition.ipynb` ~45 min, `multilingual.ipynb` ~60 min on an A100-80GB.
Note the recheck 40GB node auto-cached the 4b set at fp32 (113 GB) — the auto-cache
path inside `load_transcoder` does not take the bf16 shortcut; use `cache_transcoder`
explicitly first.

**Item 1 — fold the paper-strength ladder into `addition.ipynb` cell 11.**
Extend the strength loop from `(-1.0, -2.0)` to `(-1.0, -2.0, -3.0)` for all three
suppressions AND the magnitude low-precision readout block (m=−1 ablation / m=−2 paper
prose −1× / m=−3 paper figure −2×). Expected outputs (recheck §4.1 table above):
suppress-`_6` `8`@0.29 / `2`@0.99 / `2`@0.99; suppress-`_9` `5`@0.52 / `3`@0.96 / junk;
magnitude `5`@0.998 / `3`@0.54 / junk. Then update the three Summary rows to cite the
notebook's own numbers (currently they cite the recheck) and keep the
reproduced-at-reduced-strength framing unless item 2 changes the magnitude story.

**Item 2 — band-criterion `input_mag`, then re-run the magnitude dissociation.**
Motivation (recheck F4): `sel["input_mag"]` = the `magnitude-diag`+`mixed` catch-all,
which includes an always-on flat feature (`L0f116505`) and broadband ripples; the paper's
supernode is two clean one-operand bands (`~30`, `~59`). Spec: add band detection to
`periodicity_report` in `addition_helper.py` — compute `a_std`/`b_std` (std of a/b over
strongly-active cells, same 0.5·max threshold `s_std` uses) and label
`band-a(~mean_a)` when `a_std < 8 and b_std > 15` (symmetrically `band-b`), placed
after the mod-10 checks and before `magnitude-diag`. Threshold sanity: a ±5 band has
std≈3, mod-10 lines ≈28, uniform ≈29, an exact-46 cross stays `mixed` (its union of two
stripes has large std — correctly excluded: the paper's magnitude intervention targets
`~30`/`~59`, not the exact-value features). In cell 8 select `input_mag` =
band-labelled features only, keep the old selection as `input_mag_loose`, and run BOTH
through cell 11's ladder. Expected: `L4f148151` in the band set, `L0f116505` only in
loose; if the band set comes out empty, the 0.5 threshold clipped the band's weak
cross-arm — lower it or pick visually from `grids_input_*.png`. Acceptance: if the
ones path survives m=−2 with the clean supernode, the dissociation breakdown was
selection pollution → upgrade the magnitude verdict accordingly; if it still breaks,
the reduced-strength framing is confirmed clean.

**Item 3 — hunt the paper's actual Add Function class (stripe/band plots at the
answer position).** Motivation (recheck F3): the operator-token probe found the
*Mostly Active* signature; the paper's Add Function features (`add _9`, `add ~57`) are
one-operand conditions read where the computation happens. Spec: in cell 7 add a pool
`answer_pos_pool` = influence-top ~24 features of the ones graph **at the final
(predict-ones) position**, layers < 3·N/4 (`position_features(graphs["ones"],
[last_ones], 24, max_layer=...)`), probed with the existing ones-moment grids
(`grids_ones`; route it in `grids_for`); the panel plot then lands in
`grids_answer_pos_pool.png` automatically. In cell 8 select candidates with labels in
{`mod10-a`, `mod10-b`, `band-a`, `band-b`} (needs item 2's band labels). Acceptance:
genuine one-operand stripes/bands at the answer position = the `add _9`/`add ~57`
analogues found → flip the taxonomy row's add-function gap to reproduced (optionally
suppress them and check the lookups downstream); none → the scope-gap statement stands,
record the negative result.

**Item 4 — computed-9 screen by negative direct output weight (cell 18).**
Motivation (recheck F6): the paper's "computed 9, intermediate step" feature is defined
by its strongest *negative* direct effect on "9"; the current hunt scans label *top*
logits and cannot find it. Spec: add a small helper (mirror
`multilingual_helper.direct_logit_effect`, batched per layer via
`_get_decoder_vectors`) computing each feature's **demeaned** unembedding weight on the
`"9"` token through the final-norm scale; rank all intermediate-graph feature nodes
most-negative-first and print the top ~8 with influence and position. Acceptance:
clearly negative candidates at the final position → inspect labels/activations and
report (keeping the paper's own hedge — its evidence was weak); nothing strongly
negative → strengthen the "partial" verdict with "no suppressive computed-9 candidate
under a direct-weight screen either".

**Item 5 — Fig B5-style say-large-X readouts for the language-swap null
(`multilingual.ipynb`, section G).** Motivation (recheck F8): the paper's Fig B5
annotates say-large-multilingual ≈100%, original say-large-X 18–39%, new say-large-Y
76–105%; our null reports output probabilities only — the missing diagnostic is whether
say-large-zh moves *at all* under en→zh. Spec: extend the raw language-swap cell —
readout features = the say-big trio (`L30f27666`, `L31f11436`, `L32f100307`) + each
language's **late** language-unique final-position features (the
`lang_specific_final_features` logic on the *raw* prompts, restricted to layers ≥ N/2);
run the raw en→zh swap at 1×/3×/6× (`paper_swap` with `source_mult = 1 − s`,
`donor_mult = s`, which hits the paper's −5×/+6× at s=6) passing
`readout_layers`, and print base→steered activations at the final position
(`run_feature_intervention` already returns `baseline_features`/`ablated_features`;
`paper_swap` needs a `readout_layers` passthrough). Acceptance: if say-large-zh never
rises while the early supernodes are driven at 6×, the "causal handle sits later"
conclusion gains its mechanism (early detection does not feed the late say stage on
Qwen3-4B) → add to the language-swap Summary row; if it rises without the output
changing, that is a new puzzle — document it.

**After execution:** confirm the continuity numbers (lookup swap `8` @ ~0.715; polymer
lookups-only sums → 0%; smear widths 1.26/1.03; operand-swap crossovers
0.625×/1.0×/0.875×); refresh both Summaries and append this DEVLOG with the
per-item outcomes; `ruff check` + `ruff format --check` + `pytest` (the CI trio).

### State of the world (after the recheck session)

- Branch `verify/paper-recheck`; markdown-only notebook edits, DEVLOG + digest
  corrections, `notes/verification_recheck.md` added; ruff clean, tests pass.
- This node (40GB) now holds the 4b model + fp32-cached transcoders (113 GB — the
  auto-cache path does not take the bf16 `dtype` shortcut; re-cache with
  `cache_transcoder(..., dtype=torch.bfloat16)` if disk matters). Recheck numbers:
  `artifacts/paper_addition/4b/recheck_results.json` (node-local).

---

## Third A100 session: the five handoff items executed (strength ladder, band supernode, add-function hunt, computed-9 screen, say-large readouts) (2026-07-10)

Environment: a fresh **A100-80GB node** (Lightning studio, no Slurm), branch
`handoff/a100-80gb-queue`. Bootstrap per the handoff: `uv sync --all-groups` (ruff
clean, 98 passed / 1 skipped), minimal `.env`, both transcoder sets pre-cached at bf16
via `cache_transcoder(..., dtype=torch.bfloat16)` (57 + 91 GB), hub copies deleted,
both models pre-fetched into `.cache/models/`. One new first-run cost surfaced: the
feature-LABEL cache (`.cache/feature_labels`, separate from weights) is ~45 GB for the
4b set and downloads lazily during the first graph build (~10 min inside the first
`addition.ipynb` execution). Items implemented as commits `756b771` + `4c2780d`, both
notebooks executed end-to-end (addition twice — see item 2; zero cell errors; commit
`a955ff1`). Wall times on this node are the best yet — addition ~25 min (first run,
incl. the label download), multilingual ~35 min, addition rerun ~21 min — the bf16
transcoder caches plus a hot label cache and fast local NVMe. **bf16 parity held again on this third A100 variant**: every re-checked
committed number reproduced to the printed digit (studied pair re-selects 46+49 at
77.7%/46.0%; lookup swap `8` @ 0.7148 top-1 with the same 3 clean (9,9) donors;
polymer all-7 `1` @ 0.7467; smear widths 1.26/1.03 with `1` @ 0.8841 / `3` @ 0.9867;
introspection 6/12 carry narrations; same corpus-reuse contexts).

### Item 1 — the paper-strength ladder is in the notebook (cell 11): recheck numbers confirmed in-pipeline

All input suppressions and the magnitude low-precision readout now run m=−1/−2/−3
(ablation / paper prose −1× / paper figure −2×). The notebook's own numbers match the
recheck table exactly:

| Experiment | m=−1 (ablation) | m=−2 (prose −1×) | m=−3 (figure −2×) |
|---|---|---|---|
| suppress `_6` | **`8` @ 0.285** (9+9 story), lookups 18–45% | `2` @ 0.986, lookups ≤23% | `2` @ 0.990, all 0% |
| suppress `_9` ⚠️ wrong position (see below) | `5` @ 0.516, readouts 30–118% | `3` @ 0.961 | junk (`,` 0.39 / `""` 0.35) |
| inhibit magnitude (loose) | `5` @ 0.998, ones-path 85–116% | `3` @ 0.537, sums 0–111% | junk (flat, top `""` 0.078) |
| … low-precision readout (first digit) | `9` **0.5626**→0.376, three most-affected readouts 51–64% | `1` @ 0.230, those three → 0% | flat (`1` @ 0.074) |

The Summary's three suppression rows now cite the notebook's own ladder (they had
cited the recheck's standalone run). ⚠️ Post-scan corrections (`DEVLOG_EXTRA.md`): the
suppress-`_9` row above is the **wrong-position** measurement (§1.1 — steered at the
teacher-forced answer token; the fixed, operand-position ladder is in the addendum
below); the first-digit readout row originally quoted the ones-prompt baseline 0.99
(§1.2) and summarized 3 of 8 readouts as "bands 53–66%" (§1.4) — corrected in place
(those three most-affected readouts are `mixed`/`magnitude-diag`-class, not band-class;
the other five hold ≥97% or rise, one reaching 174.8% at m=−2).

### Item 2 — band-criterion `input_mag`: the paper-strength dissociation breakdown was selection pollution; verdict upgraded

`periodicity_report` gained band detection — but the first execution produced an
**empty band set**, the failure mode the handoff had anticipated. Measured cause (the
saved run-#1 grids): genuine bands carry a weak cross-arm in the other operand that
sits *above* the 0.5·max activity threshold (`L4f148151`'s arm inflates its b-std to
21), and the razor `b≈49` band (`L7f69527`) concentrates on ~5 residues,
hair-triggering the mod10-b concentration test (1.52 > 1.5) before any band check.
Fix (`4c2780d`): band stats use the **bright core** (> 0.7·max — 0.75 would lose
band-b) and band checks run **before** the single-operand stripe checks (safe: no
periodic stripe can have a small core std). Validated offline against run #1's saved
grids: exactly two label changes — `L4f148151` mixed → **band-a(~45)**, `L7f69527`
mod10-b(r8) → **band-b(~49)** — 23/25 unchanged; synthetics (stripes, lattices,
uniform, diagonal, equal-arm cross) keep their labels. `input_mag` is therefore the
paper-shaped **two-band supernode, one band per operand** (`band-a(~45)` `L4f148151` +
`band-b(~49)` `L7f69527` — the `~30`/`~59` analogue for 46+49; both visually verified
as bright bands with weak cross-arms), with the old catch-all kept as
`input_mag_loose` — note it grows 7 → 8 by definition (the newly-banded `L7f69527`
enters the catch-all), so its run-#2 numbers shift slightly from the committed run
(e.g. m=−2 ones flip `3` @ 0.70 vs 0.54; same phenomenology).

**Ladder outcome — the acceptance's first branch fires: the paper-strength
dissociation breakdown was selection pollution.** With the clean two-band supernode
the ones-digit pathway survives **every** strength: `5` @ 0.999 with ones-path
readouts 94–103% at m=−1, m=−2, **and m=−3** (the paper's full −2×). The loose
catch-all still breaks at m=−2 exactly as before (ones flips to `3`; sign-flipping an
always-on feature injects a global bias delta). The paper's other half is present but
shallower than Haiku's: on the first-digit graph the answer weakens progressively
(`9` 0.5626 → 0.54 / 0.51 / 0.43 across the ladder — mild; the 0.99 first quoted
here was the ones-prompt baseline, `DEVLOG_EXTRA.md` §1.2) yet the low-precision feature
readouts dip only mildly (97–101% at m=−1 → 80–105% at m=−3), where the paper
annotates its low-precision features as suppressed outright — with only two thin
bands driven, Qwen3's low-precision stage keeps most of its activation (the loose
supernode drops its three most-affected readouts to 51–64% at ablation and 0% at
m=−3 — five others hold ≥97% or rise (`DEVLOG_EXTRA.md` §1.4) — at the cost of
breaking the ones path). Magnitude verdict upgraded: the dissociation ("ones untouched, magnitude
path degraded") now holds at the paper's own strengths with the paper-shaped
supernode.

### Item 3 — the answer-position Add Function hunt: negative result (scope gap closed by measurement)

New `answer_pos_pool` panel: influence-top-24 features of the ones graph at the
predict-ones position (layers < 27), probed on the ones-moment grids
(`grids_answer_pos_pool.png`). Taxonomy (run #2, core-stat classifier): **7 lookup
lattices, 14 mixed, 1 mod10-sum, 1 magnitude-diag, and a single nominal band flag** —
`L24f133804` labels `band-b(~4)`, but the crop shows a diffuse lower-triangular
texture whose brightest cells sit at its low-b edge, not a bar on a quiet grid; the
visual check (the handoff's own prescribed arbiter) rejects it. Otherwise the pool is
the known lookup workhorses (L24f163113 (3,9), L25f90687 (5,9), L25f145046 (9,9), …),
several `a+b≈95` anti-diagonals, and diffuse high-layer textures; **no horizontal or
vertical one-operand bars** anywhere in the panel. The paper's stripe-like Add
Function class (`add _9`, `add ~57` — one-addend conditions read where the computation
happens) does **not** surface among the top-influence answer-position features of
Qwen3-4B's ones circuit; its operator-token "this is addition" (mostly-active) class
remains the only add-function-adjacent signature found. The taxonomy row's scope gap
is now a measured negative, not an instrumentation hole. (The optional
suppress-and-check follow-up is moot with no accepted candidates; `L24f133804` is only
weakly active on the studied pair.)

### Item 4 — computed-9 screen by negative direct output weight: no clean candidate; "partial" strengthened

`direct_token_weights` (batched demeaned unembedding weight through the final-norm
scale; per-feature identical to `multilingual_helper.direct_logit_effect`) screened
all 1137 features of the `assert (4 + 5) * 3 ==` graph. The negative tail exists —
min w9 = −0.388 vs max +0.476, median 0.000 — and includes two final-position
features (`L30f12843` w9=−0.39, act 1.5, influence 0.0009; `L32f149033` w9=−0.27,
act 22.1, influence 0.016) plus several at the intermediate `4 + 5` positions
(`L31f4652@p4` −0.30, act 26). But none has the paper's computed-9 character: all
top-logit labels are non-numeric junk (` Yay`, `ham`, `穿戴`, quote-fragments), and
influences are marginal (≤0.016). With the instrument bias removed (the old scan
could not see suppressive features by construction), the "partial" verdict is
strengthened: **no suppressive computed-9 candidate under a direct-weight screen
either** — Qwen3-4B computes 27 without a detectable flag-the-intermediate-9 feature
in the pruned graph.

### Item 5 — Fig B5-style say-large readouts: the language-swap null gains its mechanism

`paper_swap` gained a `readout_layers` passthrough; `lang_specific_final_features`
gained `raw`/`min_layer_frac` (late ≥ L18 language-unique final-position supernodes —
the say-large-X analogues; they populate at L29–35 on the raw prompts, acts 15–200).
The raw en→zh swap at 1×/3×/6× (source 0×/−2×/−5×, donor +1/+3/+6 — the paper's
−5×/+6× endpoints at 6×) reads out the say-big trio + the late en-unique and
zh-unique supernodes on the same perturbed forward. Result, vs the paper's Fig B5
annotations (say-large-multilingual ≈100%, old say-large-X 18–39%, new say-large-Y
**76–105%**):

- **say-large-zh never comes online**: 9/12 late zh-unique features read exactly
  0.00 at baseline and at every strength; one baseline-zero member (`L31f104452`)
  wakes to ~4 — an order of magnitude below the active scale (15–70); of the two
  baseline-active members, `L31f21719` wobbles (15.3 → 20 / 9.5 / 12.8) and
  `L29f56902` falls 81% at 6× (4.7 → 0.9). (Tally corrected per `DEVLOG_EXTRA.md`
  §2.1 — originally miscounted as "10/12 at every strength".)
- the say-big trio's anchor `L30f27666` holds ≈100% at 1×/3× (28→27–29 throughout)
  while the trio mean-of-ratios reads 90.4/84.7% (scan §2.4 wording tightened, fourth
  session); at 6× two of three degrade (29%/55%; mean 60.2%) — the familiar over-drive
  erosion;
- the late **en**-say supernode only partially weakens at 6× — by thirds: 4/12 at
  47–55%, 4/12 at 66–82%, 4/12 at 90–105% (mean 74.5%; scan §2.4 wording tightened,
  fourth session) — directionally the paper's "old say-large-X suppressed" but far
  shallower;
- the output **consolidates on English**: `large` 0.79 → 0.86 → 0.94 (the committed
  sweep's strengthening, now with its internals visible — the drive suppresses
  competitors like `big` 0.20 → 0.05 rather than recruiting Chinese).

Acceptance branch taken: say-large-zh never rises while the early supernodes are
driven at the paper's full strength ⇒ **the "causal handle sits later" conclusion
gains its mechanism** — on Qwen3-4B, early language detection does not feed the late
say stage, so swapping it cannot move the output language. Added to the language-swap
Summary row.

### Same-session addendum 1 — Fig B3/B4 supernode %-readouts (mean of per-feature ratios)

Recheck F8's remaining soft note ("Fig B3–B5's supernode %-readouts were not reproduced
as readouts anywhere") is now closed for all three swap figures. New
`multilingual_helper.supernode_readout_pct`: each feature's ratio ``steered / reference
× 100`` is computed **individually, then averaged** (mean of ratios — a big feature must
not drown the others; NOT the ratio of summed/averaged activations); ~0-reference
features are excluded and counted. Reference = the recipient-prompt clean activation
(`baseline`) or the stored donor-/own-prompt activation (`stored`). The operation,
operand and raw-operation cells rerun each swap at the paper endpoint and at each
language's crossover with these readouts; the language-swap readout cell prints
supernode means. **Reading the numbers** (per the pre-report scan, `DEVLOG_EXTRA.md`
§2.2/2.3, folded into the design): encoder-side readouts are blind to a feature's
own-layer decoder delta, so steered supernodes (sources, injected donors) report the
network's *propagated response*, not the commanded steer; the paper-comparable
annotation numbers come only from supernodes **disjoint from the steered sets** — for
the operation swaps that is the new `downstream_say_large` supernode (late
language-unique say features + unsteered say-big members; the say-big trio itself sits
inside the steered source there), for the operand swap the (unsteered) say-big trio and
the steered-feature-excluded say-cold supernode.

Results vs the paper's annotations:

- **Operand swap (Fig B4: say-large ≈0–12%, say-cold recruited, upstream ≈100%)** — at
  the +1.5× endpoint the downstream pattern lands: say-large (trio, genuinely
  downstream here) **0.0% / 0.0% / 0.0%** (2.3–8.9% already at the crossovers) in the three languages vs the paper's ≈0–12%; the
  say-cold supernode reaches **110% / 148% / 65%** of its own-prompt level ("recruited",
  with the caveat that it contains task-general members pre-active on the recipients).
  The final-position operation proxy falls to 41–58% where the paper's
  upstream stays ≈100% — partly definitional (it retains say-side members that
  legitimately drop when large→cold).
- **Operation swap (Fig B3: upstream ≈100%, downstream say-large ~11/10/13% and
  0/0/20%)** — the disjoint `downstream_say_large` supernode reads **69.0% (EN) / 1.0% (FR) / 83.9% (ZH)** at the
  raw format's full 6× (chat: 74.4 / 6.5 / 40.6% at 6×); the trio row (inside the steered source)
  reads 11.2/0.0/46.0 raw @6× but mixes commanded steer with propagation and is not
  the paper's annotation. `upstream_operand` is 99.8–100.2% everywhere — matching the
  paper's ≈100%, though causally guaranteed under frozen attention patterns (no
  evidential weight). Donor rows: propagated response 89–106% of donor level at
  6× (their lowest-layer members necessarily read ~0% of stored).
- **Language swap** — supernode means compress the item-5 finding: say-large-zh stays
  **≤7.6%** of its own-prompt level at every strength (paper's new-language target:
  76–105%) vs say-big 90/85/60% and en-say 99/94/74%.

### Same-session addendum 2 — eager decoders: the notebooks' load-bearing step removed

Profiling the runs (artifact-mtime timeline) showed the dominant cost was not the
graphs or the grids but the **intervention volume × the lazy-decoder clean baseline**:
every `run_feature_intervention` call rebuilds `capture_constants` + a full
LocalReplacementModel forward, and with `lazy_decoder=True` that forward re-streams all
36 decoder matrices (~28 GB) — ~9.3 s per call, ~195 calls per multilingual run ≈ 30 of
its ~50 minutes. Both notebooks now load the 4b transcoders with **`lazy_decoder=False`**
(cell 1). Pre-flight measured: one `paper_swap` 9.3 s → **0.18 s (~52×)**; 8000-node
graph-build peak **68.0 GiB** — inside the A100-80GB with headroom. Measured
end-to-end: `multilingual.ipynb` ~50 min → **14 m 40 s**, `addition.ipynb` ~21 min →
**17 m 50 s** (its grid probes are encode-only and keep their ~10 min). The §H 8b load
deliberately stays lazy: the overlap analysis is **encode-only** (it only reads feature
activations via `transcoder.encode()`; nothing is written back through `W_dec`), so
laziness costs it nothing — and eager 8b decoders (+encoders + model ≈ 107 GB) would
not fit the card. Determinism check: the chat-format B3/B4 readout JSONs from the lazy
and eager runs are identical to the printed digit. CLAUDE.md's performance note
updated with the measured numbers.

### Same-session addendum 3 — pre-report scan (`DEVLOG_EXTRA.md`) response

An independent adversarial scan (recorded in `DEVLOG_EXTRA.md`, no code changed there)
landed mid-session; its findings were addressed as follows (both notebooks re-executed
end-to-end afterwards, zero cell errors):

- **§1.1 (HIGH) — fixed + re-measured.** `digit_token_positions` collected the
  teacher-forced answer digits into `b_digits`, so `suppress_9` steered the
  predict-ones position (10) instead of the operand's ones digit (8), and the
  `input_b` pool mixed in answer-position nodes. Fix: digit collection stops at `=`
  (regression-tested against the real tokenizer and in
  `tests/test_notebook_helpers.py`). Re-measured operand-position ladder:
  suppress-`_9` m=−1 → **`8` @ 0.711**; m=−2 (paper prose −1×) → **`8` @ 0.678**;
  m=−3 (figure −2×) → `2` @ 0.347 (readouts: all 4 sums → 0% from m=−1 on; lookups
  ≤10% except `L24f163113` surviving at 54% → 42% → 4%). The corrected experiment is
  *stronger* than the wrong-position one had suggested: a crisp ones-digit
  substitution (`5`→`8`) that **survives the paper's −1×** — where suppress-`_6`'s
  9+9 story collapses to `2` — degrading only at −2×. Both operand suppressions now
  land on `8`, consistent with the surviving multi-pair lattices (receptive fields
  include (9,9)→8); like the paper's own 91 ("not 92 — grain of salt"), the
  substituted digit resists simple numerology. `input_b`'s pool now draws from the
  true operand positions {7, 8}. Selection stability: `input_9` keeps its 6 features
  and the band pair is unchanged; the loose catch-all swaps one member (the
  answer-position artifact `L0f108251` exits, position-7 `L0f147779` enters), so its
  ladder numbers shift once more (ones flip at m=−2 now `3` @ 0.535; three
  most-affected low-precision readouts 52–64% at ablation, first digit
  0.5626 → 0.362) — same phenomenology, and the band-supernode rows are untouched. Historical tables carry ⚠️
  notes; the Summary row is rewritten from the fixed run.
- **§1.2–1.5, §2.1 (MED/LOW) — corrected in place.** First-digit baseline 0.99 →
  0.5626 (ones-prompt value had been copied); "sum feats → 0%" and "deep 51–64%
  suppression" rewordings (partial, mixed-sign readouts); answer_pos_pool "15 mixed" →
  14 mixed + the visually-rejected band flag; the zh-unique tally 10/12 → 9 always-zero
  + 1 waker + 2 baseline-active (one falling 81% at 6×).
- **§2.2/2.3 (MED) — readouts redesigned** (see addendum 1): `downstream_say_large`
  added as the paper-comparable annotation; trio/source/donor rows relabeled as
  propagated response; `supernode_readout_pct` docstring now states the encoder-side
  blindness to own-layer steers.
- **§3.1 (HIGH) — documented + quantified; semantics unchanged.** The module header no
  longer claims unqualified circuit-tracer faithfulness: propagate mode is documented
  as clean-anchored fixed-delta steering (exact vs the clamp for single-layer/uncoupled
  interventions; deliberate protocol for coupled stacks), and the
  `run_feature_intervention` docstring's false "defaults to the last steered layer"
  sentence is fixed (the default is `None` = propagate). Quantification on the
  flagship multi-layer cases (this run's own selections): lookup swap —
  propagate **`8` @ 0.7148** (the committed, paper-margin result) vs constrained-to-l_max **`1` @ 0.7321** — the swap does not land (`8` drops out of the top-4); polymer 7-feature
  suppression — propagate `1` @ 0.7467 vs constrained `1` @ 0.3525 (same flip, half the mass). The protocol choice is material on coupled stacks: the injected donors need the downstream recompute to reach the answer, so the notebooks' results — including the paper-margin swap match — are propagate-mode results and are now described as such everywhere. (The paper's own range-choosing step exists as `sweep_patch_end_layer` for anyone wanting the constrained picture per case.)
  **Decision (same day, after verifying biology.html):** the paper's dives all run
  under constrained patching ("Our interventions in this paper use the 'constrained
  patching' technique" — clamped prior to a chosen intervention layer), so the
  reproductions will adopt it: constrained numbers with a swept, stated ℓ become the
  headline results; propagate-mode results remain as the no-pinning robustness
  variant. Queued for the next session — the executable spec (helper plumbing,
  ℓ-sweep policy, scope, pinned-readout caveat, presentation rule, expected deltas)
  is `DEVLOG_EXTRA.md` §0 row 3.1's Status cell.
- **§4.1 (MED) — fixed.** `verify_intervention.py` now exits 1 on FAIL.
- **§4.3 (MED) — closed.** New `tests/test_notebook_helpers.py` puts the
  verdict-carrying helper logic under pytest (position parsing incl. the §1.1
  regression, the band-criterion classifier, `direct_token_weights` vs the reference,
  `supernode_readout_pct` mean-of-ratios/refs/skips) — 110 tests pass; CI's lint and
  format steps now also cover `notebooks/` and `verification/`.
- **Left as recorded findings (no code change):** §3.2 (attribution "all"-mode sink
  semantics — library/serve exposure only; notebooks unaffected), §3.3 and §5.x
  periphery items, §6 housekeeping. These remain documented in `DEVLOG_EXTRA.md` for
  a future hygiene pass.

### State of the world (after the third A100 session)

- Branch `handoff/a100-80gb-queue`: `756b771` (items as code), `4c2780d` (band-core
  fix), `a955ff1` (executed notebooks + refreshed Summaries), `1ee7ce5` (DEVLOG),
  then the same-session addenda commits (Fig B3/B4 readouts + eager decoders +
  pre-report-scan fixes with both notebooks re-executed). ruff clean, **110 tests
  pass** (notebook-helper suite added), CI lint/format now covers `notebooks/` +
  `verification/`, zero cell errors on every execution. All five HANDOFF items are
  closed. **Queued: the constrained-patching pass** (scan §3.1's follow-through —
  spec in `DEVLOG_EXTRA.md` §0 row 3.1); the remaining recorded-but-unfixed scan
  findings (`DEVLOG_EXTRA.md` §3.2, §3.3, §5, §6) are library/serve/housekeeping
  items outside the reproductions.
- Node state: bf16 transcoder caches (57 + 91 GB) + ~45 GB feature-label cache under
  `.cache/`; both models under `.cache/models/`; artifacts regenerated under
  `artifacts/paper_{addition,multilingual}/{4b,8b}/` (node-local, gitignored).
- Continuity extra: the paper-faithful polymer lookups-only suppression, re-run
  standalone on this run's own selections — 3 active lookups suppressed at m=−3 →
  `5` @ 0.9824 → **`1` @ 0.7708** with **all 4 sum features at 0%** of baseline
  (recheck's Fig A5 panel 2 resolution intact).
---

## Fourth A100 session: scan-fix verification + the constrained-patching pass (2026-07-11)

Environment: a fresh **A100-80GB node** (Lightning studio, no Slurm), branch
`fix/constrained-patching-pass`. Bootstrap per the handoff recipe: `uv sync
--all-groups`, minimal `.env`, both transcoder sets pre-cached bf16 (57 + 91 GB, hub
copies deleted), both models pre-fetched; the ~45 GB feature-label cache re-downloaded
lazily during the first graph build (slower on this node — the first `addition.ipynb`
execution took ~50 min wall, most of it label downloads; `multilingual.ipynb` reused
the cache).

### Step 1 — every ✅-marked scan fix independently re-verified

Each DEVLOG_EXTRA §0 row marked fixed was re-checked against the code, the tests, the
CI config, and the committed notebook outputs (not the prose): §1.1 (digit collection
stops at `=`; 3 regression tests; ladder numbers `8`@0.7111 / `8`@0.6784 / `2`@0.3472
match cell 11's committed outputs digit-for-digit), §1.2 (0.5626 baseline quoted;
steered 0.5405/0.5072/0.4275 match), §1.3 (all four sums do read 0% from m=−1 at the
corrected position), §1.4 (51.6/63.5/57.8 + five ≥95.8% + the 176.3% row match),
§2.1 (tally re-derived feature-by-feature from the committed readout printout: 9
always-zero + 1 waker to 3.64 + L31f21719 wobbling + L29f56902 4.69→0.90 = −81%),
§2.2 (disjoint `downstream_say_large` present; FR-only collapse 1.0% raw / 6.5% chat
vs EN/ZH 69/84% raw match cell 22/8 outputs; the trio row correctly relabeled), §2.3
(docstring + prose state the encoder-side blindness), §4.1 (`sys.exit(1)` on FAIL),
§4.3 (12 helper tests in CI; lint/format cover `notebooks/` + `verification/`), and
the 3.1 doc-half from `9d61004` (module header scoped, docstring default corrected).
**All hold.** One wording reconciliation: the Summary's donor range "67–106%" is the
chat∪raw union while this DEVLOG's "89–106%" is raw-only — both faithful to the
outputs. Two §2.4-class prose-generosity items were tightened this session in the
multilingual Summary/DEVLOG (trio "≈100% at 1–3×" → measured means 90/85%; en-say
"≈half drop to ~50%" → thirds at 47–55 / 66–82 / 90–105, mean 74.5%).

### Step 2 — the §3.1 decision executed: constrained patching is the headline protocol

Plumbing (`bc71f40`, +6 tests → 116 pass): `patch_end_layer` passthrough on
`steer_and_report` / `paper_swap` / `paper_swap_sweep` / `run_graft` and per-job in
`run_swap_sweeps`; `choose_patch_end_layer` in both helpers implements the paper's
recipe — sweep ℓ ∈ [l_max, n_layers−1] at the paper's endpoint strength and take the
most effective end layer on the target-token metric (suppressions: most-suppressive
logit via `best_end_layer`; injections/swaps: max target probability via the new
`LayerSweepResult.most_promoting_end_layer`); readout rows at layers ≤ ℓ are pinned by
the protocol and dropped (`n_pinned` in `supernode_readout_pct`, `pinned: True` in
`steer_and_report`). Both notebooks run every scoped intervention in both modes
(`cf9356b` code, executed end-to-end with zero cell errors); the chosen ℓ is stated
next to every constrained result; Summaries present constrained as the headline and
propagate as the no-pinning robustness variant.

**Continuity (propagate mode, fourth A100 variant):** exact to the printed digit again
— studied pair re-selects 46+49 (77.7%/46.0%); suppress-`_9` ladder 0.7111/0.6784/
0.3472; smears 1.26/1.03 (`1`@0.8841 / `3`@0.9867); polymer all-7 `1`@0.7467 and
lookups-only `1`@0.7708 with all 4 sums at 0% (that run is now an in-notebook cell,
not a recheck-only standalone); lookup swap `8`@0.7148; multilingual crossovers
0.625×/1.0×/0.875× operand, 3×/1×/2× chat-op, 3×/1×/1× raw-op (with the same p_exp peaks 0.221→…/0.730/0.911 raw), language nulls in both formats, the raw en→zh readout means 90.4/98.6/7.6 → 60.2/74.5/5.4, and the §H scale table (8b 0.132/0.098/0.082 > 4b 0.107/0.089/0.077, set sizes 935/1206); IoU/scale table unchanged.

### Addition under constrained patching (chosen ℓ next to each)

| Experiment | l_max | ℓ (sweep @ paper strength) | propagate (robustness) | constrained (headline) |
|---|---|---|---|---|
| suppress `_6` | 7 | 15 | `8`@0.285 / `2`@0.986 / `2`@0.990 | `3`@0.82 / `3`@0.88 / `3`@0.36 — crisp at paper strengths, never the 9+9 `8` |
| suppress `_9` (position-fixed) | 6 | 9 | `8`@0.711 / `8`@0.678 / `2`@0.347 | `5` intact @0.97 at ablation; `2`@0.92 at −1×; `2`@0.65 at −2× |
| inhibit magnitude (band pair) | 7 | 9 | ones 0.999 all m; first digit 0.5626→0.54/0.51/0.43 | ones 0.999 all m (readouts 96–102%); first digit →0.557/0.507/0.477, low-prec 80–106% |
| inhibit magnitude (loose) | 7 | 11 | ones breaks at m=−2 (`3`@0.535) | **ones survives every strength** (0.999; readouts 82–115%); first digit →0.42/0.28/0.12, three most-affected low-prec 27–65% at −2× |
| neg-steer lookup (m=−3) | 25 | 25 | flips `1`@0.8841, width 1.26 | flips `1`@0.956, width 1.09 |
| neg-steer sum (m=−3) | 34 | 35 | flips `3`@0.9867, width 1.03 | **smears** — width 4.10 (`4`@0.39, `3`@0.24, `9`/`7`/`6` ≈8% each) |
| polymer all-7 (m=−3) | 34 | 35 | `1`@0.7467 | `4`@0.45 (two-point sweep picks the pure-direct ℓ=35; readouts all pinned) |
| polymer lookups-only (m=−3) | 25 | 25 | `1`@0.7708, sums 0% | `1`@0.8315, sums 0% — the Fig A5 panel-2 claim, protocol-exact |
| lookup swap (9,9) donors | 25 | 29 (best of a full curve) | **`8`@0.7148 top-1** (paper's margin) | **no landing layer**: p(`8`) = .032/.043/.072/.064/.087/.068/.077/.057/.019/.013/.017 over ℓ=25..35; best gives `1`@0.39 with `8` third |

Readings. (1) The **magnitude/ones dissociation gets stronger** under the paper's
protocol: it now holds at full paper strength for *both* supernodes — the loose
catch-all's propagate-mode breakage at m=−2 was the unpinned cascade of the
sign-flipped always-on `L0f116505` through the recomputing early layers, which
constrained patching cuts by construction. Pinning also lets the magnitude side
degrade deeper (first digit to 0.12 at −2× with low-prec readouts 27–65%) without
touching the ones path — closer to the paper's annotation pattern than either
propagate run. (2) The **paper's sum-side smear appears under the paper's protocol**:
at the swept ℓ=35 the negative sum steer spreads the ones digit over width 4.10 (the
paper: "smears the result out to a wider band"), where propagate had shown a sharp
re-sharpened flip; the lookup side keeps flipping under both protocols (the paper's
smear-over-~5 for lookups still does not appear). (3) The **9+9 story and the
swap's paper-margin landing are propagate-specific**: under constrained patching the
input suppressions substitute different digits (`3`/`2`), and the (9,9) donor
injection never drives `8` past 8.7% at any end layer — on Qwen3-4B the swap's causal
route needs the within-range recompute that pinning freezes (Haiku's swap landed
under its constrained protocol; ours needs propagation — a real model/protocol
difference now measured rather than hidden).

### Multilingual under constrained patching

The three swap families under the paper's protocol (ℓ stated per language; the
constrained strength sweeps and endpoint readouts live in the new `constr-*` cells):

| Swap | l_max (selection) | ℓ chosen | propagate (robustness) | constrained (headline) |
|---|---|---|---|---|
| operation, chat + raw | **35** — the influence-top final-position supernodes reach the last layer | 35 (forced; single-point sweep) | crossovers 3/1/2× chat, 3/1/1× raw; echo-synonym top-1 in all three raw languages | pure direct effect, every %-readout pinned; the direct push alone still lands the synonym mode mid-sweep (raw ZH 小 @ 0.95 at 1×, raw FR `pet` @ 0.51 at 2.5×, raw EN `small` @ 0.997 at 6×; chat EN's designated `tiny` reaches 0.221 vs 0.007 propagate) before endpoint junk |
| operand | 28/28/27 | 28/28/27 (= l_max, the sweep's best) | crossovers 0.625/1.0/0.875×; cold / f / 冷 top-1 | **no crossover in any language** (p_exp ≤ 0.032); at +1.5× EN echoes the injected `hot` @ 0.74 instead of computing its antonym, FR/ZH keep baseline answers; say-cold not recruited (0–45% of stored vs 65–148% propagate); say-big 16–66% |
| language, chat + raw | 10–11 (early detection supernodes) | swept 12–25, choice vacuous | p_exp = 0 at every strength, both formats | **p_exp ≈ 0 at every (ℓ, strength)** — max 0.0004 over the full end-layer sweep in all six direction×format combinations; constrained Fig B5 readouts (raw en→zh, ℓ=12): say-large-zh ≤ 10.7% of stored at every strength, say-big/en-say 60–98% |

Reading: constrained patching **interacts with the selection style**. The paper's
hand-curated supernodes are early/mid concept groups, so its protocol left Haiku's
swaps room to recompute; our supernodes are influence-tops that reach L27–35, so the
same protocol pins most of the network — the operation swap degenerates to a
(surprisingly effective) direct-logit push, the operand swap — the cleanest propagate
reproduction — cannot complete the antonym recomputation inside the pinned range, and
the language-swap null upgrades to protocol-complete: no end layer, strength,
direction, or prompt format moves the output language via early detection features.
The multilingual verdicts keep their propagate basis with the constrained picture
stated alongside (Summary rows updated in place).

### State of the world (after the fourth session)

- Branch `fix/constrained-patching-pass`: `bc71f40` (plumbing + tests), `cf9356b`
  (notebook code), `5224c99` (executed notebooks + Summary updates), plus this
  DEVLOG + DEVLOG_EXTRA close-out commit. ruff clean, 116 tests pass, zero cell
  errors. Wall times: addition ~34 min (~20 min of it the label re-download),
  multilingual ~15.5 min (warm caches; the added ℓ-sweeps cost ~1–2 min each run).
- DEVLOG_EXTRA §0 row 3.1 is closed; §3.2/§3.3/§5.x/§6 remain recorded-open
  (library/serve/housekeeping, outside the reproductions).
- Node state: bf16 transcoder caches (57 + 91 GB), ~45 GB feature labels, both models
  under `.cache/`; artifacts regenerated under `artifacts/paper_{addition,multilingual}/`.

### Same-day addendum — intervention-method audit (solidify before any protocol cleanup)

Prompted by the constrained-only-cleanup question: our constrained patching was
re-audited **line-by-line against the pinned circuit-tracer source** (the paper
authors' reference implementation) and the methods paper's own description. Mechanics
are identical: MLP outputs pinned to clean inside [0, ℓ] with deltas anchored on
*clean* activations (`value − clean`), attention patterns frozen everywhere with live
V/O (so within-range attention responds linearly — circuit-tracer's reading of the
paper's stricter "run forward from the last layer of the range" prose; we follow the
implementation), LayerNorm frozen iff the range covers all layers, real model after ℓ.
Two deliberate divergences, both safety-positive: we **raise** on ℓ < l_max where
circuit-tracer **silently drops** out-of-range interventions (`intervention_hook` only
fires for in-range layers), and steering conventions differ per the documented
m/M/value mapping. `verify_intervention.py` extended from one parity case to three —
single m=−2, a two-layer coupled stack with ℓ above the top steer, and a `value=`
donor injection on a clean≈0 recipient feature — **3/3 PASS at the noise floor**
(max Δ-diff ≤ 3e-5 on ‖Δ‖ 37–130, cosine 1.00000; A100 fp32). Every intervention
primitive the reproductions use is therefore cross-verified in the headline
(constrained) mode. Reading of the fourth-session constrained failures stands
strengthened: **not a method bug** — the paper's protocol paired with our
influence-top supernode *selections* (which reach L27–35) leaves no recompute room;
the fair paper-style test, queued for when the reproduction work resumes, is
re-selecting the swap supernodes the paper's way (early/mid hand-curated concept
groups) and re-running the constrained ℓ-sweep.


---

## Fifth session: reproduction v2 — reviewable semantic supernodes + constrained-only (addition live, multilingual parked) (2026-07-11)

Branch `repro/constrained-supernodes` (same A100-80GB node). Motivation: the fourth
session showed constrained patching is method-exact but starved by influence-top
supernode selection; the paper's own supernodes are small hand-curated concept
clusters, and its membership is **activity-based** (20/27 members active vs 10/27 in
its pruned graphs — biology digest).

### The supernode pipeline (durable workflow)

1. **`notebooks/build_supernode_inputs.py`** (GPU, once per config): persists the 13
   pruned graphs as JSON (nodes with influence + labels + activation examples — the
   explorer HTML had dropped influence), all-feature operand grids (3 single-pass
   probes, `grids.npz`), polymer ones-moment activations, and a manifest
   (prompts/positions/answers/git sha). Artifacts numbers-layer is now committed
   (gitignore carve-out: JSONs/PNGs/npy in, heavy HTMLs/graph dumps/npz out).
2. **`notebooks/build_supernodes.py`** (CPU): scans ALL pruned-graph feature nodes and
   proposes paper-named supernodes by semantics — **both label sides** after a user
   review catch (output side = top logits, what a feature promotes; **input side =
   the peak-activation tokens of its examples**, what it fires ON; sides recorded per
   member as provenance). Addition membership is by operand-grid receptive-field
   class (+ on-pair fraction evidence); ≤6 members ranked by activation with
   runners-up in `overflow`; disjointness enforced; influence recorded as evidence
   only. Auto-flags: script-mismatch say-large matches, <30% on-pair lookups.
   Reviewer explorer HTMLs per graph with proposed groups pre-loaded.
3. **Review gate**: files ship `approved:false`; `load_supernodes` refuses unapproved/
   rejected/overlapping files. Addition was reviewed by delegation (user waived the
   manual pass for the grid-principled selection): one action — lookup L33f109892
   rejected (20% on-pair) → L24f99664 promoted (63%); `review_log` in the file.
4. **Notebooks load the reviewed file** and the persisted graph dumps (interventions
   run against literally the graphs the review saw; `addition_input_ids` extracted so
   the load path reconstructs identical tensors).

Input-side lesson (user catches, both fixed in-session): top-logit-only matching had
(a) starved the paper's INPUT supernodes — antonym went 3→6 members (the paper's own
size) and small 2→6 once example-peak matching landed — and (b) produced a false
"Qwen3 has no synonym-operation features" claim: a both-side any-position scan finds a
clean cluster firing on ⟦synonym⟧/⟦synonymous⟧ at the operation-word token, now the
paper-faithful `synonym (operation)` donor (say-answer demoted to alternative).

### Addition v2 (constrained-only, reviewed supernodes; zero cell errors, ~13 min)

| Experiment | ℓ (swept) | v2 result | vs paper |
|---|---|---|---|
| suppress `_6` | 16 | m=−1 nothing (`5`@0.996); −1× **`3`@0.961**; −2× `3`@0.736; sums→0% from −1× | crisp substitution at paper strength; no 9+9 numerology (paper warns on numerology) |
| suppress `_9` | 9 | m=−1 nothing; −1× **`2`@0.940**; −2× `2`@0.899 | same shape |
| inhibit magnitude (band pair) | 7 | ones `5`@0.999 ALL strengths (readouts 96–106%); first digit 0.526/0.478/0.422, low-prec supernode 87–97/68–91/**42–81%** | dissociation at full paper strength |
| neg-steer lookup / sum (m=−3) | 25 / 35 | lookup flips `1`@0.84 (width 1.38); sum **smears width 5.26** (`1` .31/`3` .21/`2` .16/`4` .11) | sum-side smear matches; lookup-side still flips |
| polymer lookups-only (m=−3) | 25 | `1`@0.83, **all six sums → 0%** | Fig A5 panel 2 protocol-exact |
| lookup swap (6 reviewed (9,9) donors) | swept 27–35 | **p(`8`) ≤ 0.067 at every ℓ** (best ℓ=31 → `1`@0.267) | does NOT land — clean model difference (selection/donor/protocol all controlled; v1 propagate `8`@0.715 documents the needed route) |

Notable protocol shape: under constrained patching with these supernodes, **ablation
(m=−1) does nothing** for the input suppressions — the effects begin exactly at the
paper's stated strengths. Runtime: graphs load from dumps (~6 min saved), ~13 min
end-to-end.

### Parked (multilingual) — state for resumption

Proposals emitted and pushed (`multilingual_4b.json`, approved:false): antonym 6 (both
sides), synonym (operation) 6 + say-answer alternative, small 6, hot 6 (+51 overflow),
say-large trio+3, per-language say-large with script-mismatch flags, detectors =
off-graph v1 fallback (RECORDED FINDING: pruned raw graphs keep NO early nodes at the
quote position), raw antonym = chat-derived fallback (raw graphs keep almost nothing).
Open decisions: fallback approval, the flagged members, and optionally re-dumping raw
graphs at node_threshold 0.95 (`--node-threshold` knob added; user deferred — the
paper's activity-based membership precedent argues for it or for activation-universe
selection). One working-tree staleness incident (editor autosave reverted the
supernode JSON; committed version was correct; restored) — reload editor tabs of
files under `notebooks/supernodes/` before editing.

### State of the world

- Branch `repro/constrained-supernodes`, pushed. 121 tests green (loader suite added),
  ruff clean, both --check validators OK. addition_4b.json approved; multilingual
  parked unapproved. Node caches warm (weights + labels + supernode inputs).

### Same-day addendum — kind coverage, evidence figure, notebook cleanup

User review vs the paper's own graph inventory drove three upgrades. (1) Two node
KINDS existed on Qwen3 but had no classifier class: `exact-cross(V)` (row+column union
at one value — the paper's `36`/`59` inputs; 3+6 members found, and several former
input-lattice members correctly migrated there) and `region(~a,~b)` (2-D localized
blobs — the wide/narrow magnitude-lookup class; 6 members on the first-digit circuit,
1–2 visually clean + several diffuse washes worth a review pass). (2) The add-function
negative became a LIVE scan result: an `add function (hunt)` supernode with a
pair-consistent rule — the unconstrained scan's 13 hits were off-pair junk textures,
the pair-consistent 2 are junk-labeled mod10-b(r9) shapes (flagged) — effectively
negative, now reviewable in the file and visible in the evidence figure.
(3) Readability: the seven v1 influence-pool panels and their taxonomy builder are
retired (markdown 20.7k → ~11k chars; runtime ~13 → ~8 min; stale artifacts git-rm'd:
7 panel PNGs, taxonomy.json, propagate-era interventions.json); the new
`grids_supernodes.png` shows every reviewed member's operand grid, one supernode per
row — the membership evidence in one figure. Delegated review decisions moved INTO the
selector (`REJECTED_MEMBERS`/`APPROVED_TASKS`, re-applied on every emit) so re-runs
reproduce reviewed state. Re-executed clean (~8 min): suppress-6 [ℓ=16] `3`@0.929 at
−1×, suppress-9 [ℓ=9] `2`@0.963, magnitude/smears/polymer unchanged, swap still lands
nowhere (p(`8`) ≤ 0.067) — the purified sets sharpened, none of the verdicts moved.

## Sixth session (2026-07-11, same day): multilingual selection is EXPORT-DRIVEN and reviewed

The user set the selection principle: review pages in the explorer UI are the ONLY
supernode-selection channel for multilingual — seeds proposed by the semantic scan,
the human adjusts groups on the pages, "Export groups" JSONs are ingested verbatim
(`build_supernodes.py --from-exports`) into the authoritative file. Manual-override
channels, off-graph fallbacks, and the scan-written multilingual file were removed.

### What shipped

- **Paper-vocabulary supernodes** (user decision: fully faithful; admit absences):
  lang-specific `opposite (lg)` / `quote (lg)` / `say large (lg)`; multilingual
  `antonym / synonym / small / hot / say large / say small / say cold (multilingual)`.
  The group NAME is the merge key across pages: identically-named groups union into
  one supernode (chat + raw prompts alike) with per-graph acts/positions per member —
  interventions steer the whole union. `large/cold (multilingual)`, `say hot`: no
  defensible members found → admitted absent (ROLE_BY_NAME entries exist if later found).
- **Dumps by design**: chat graphs at 0.8 (circuit-tracer default; 461–746 feature
  nodes), raw graphs at 0.95 (at 0.8 they starve — zero quote-position nodes for the
  detectors). `raw_hot_en` added: the paper's EXACT donor prompt (all paper prompts
  are raw completions; chat forms are our Qwen3-instruct adaptation — an earlier
  docstring claimed the opposite rationale and is fixed). Pages named
  `review_chat_*` / `review_raw_*`; 10 multilingual + 3 addition.
- **Review completed (2026-07-11)**: the user reviewed all 10 pages and accepted the
  seeds as-is; exports were materialized verbatim from the approved pages' embedded
  groups and ingested → `notebooks/supernodes/multilingual_4b.json` (16 supernodes,
  approved: true, selection: explorer-export; source exports committed under
  `notebooks/supernodes/exports/`).
- **Strict disjointness** (global by (layer, feature), the paper's semantics) caught
  three real seed overlaps; deterministic resolutions now in the seed rules:
  quote beats opposite (the paper's quote-features "track language via other words");
  say-large multilingual/lang-specific exclusion spans chat+raw scans; op-word
  features firing for ≥2 languages are language-ambiguous and seed neither. Per-PAGE
  groups stay ≤6; the reviewed cross-page union may exceed (recorded `max_members`;
  antonym 7, synonym 8 — paper: 6/6).
- **Fixed en route**: seeded groups with a `member-positions` marker were silently
  dropped by the page renderer — the antonym + synonym groups (operation swap's
  source and donor!) were missing from every page before this session's re-emit.

### NEXT (phase B, next session)

1. Rewire `multilingual.ipynb` constrained-only from the loaded file: jobs built from
   roles (antonym −5×/m=−6 + synonym value=+6× donor at the recipient's operation-word
   token; small −0.5×/m=−1.5 + hot value=+1.5×; quote swaps −5×/+6×), per-experiment
   `sweep_patch_end_layer`, %-readouts from say-* supernodes at layers > ℓ.
2. Addition-style prose cleanup + re-execution (~15 min GPU), stale propagate-era
   multilingual artifacts git-rm'd, Summary rewritten as v2 verdicts, DEVLOG.
3. Later: revisit addition supernodes via the same export workflow (user intent).

Resume commands: `python3 -m http.server 8000 --bind 0.0.0.0 --directory artifacts`
(review pages); heavy dumps live on the studio disk and regenerate deterministically
via `notebooks/build_supernode_inputs.py` at the manifest's recorded sha/thresholds
(~35 min GPU) if ever lost.

## Seventh session (2026-07-12, fresh studio): multilingual v2 executed — constrained-only from the reviewed file

Studio disk from the sixth session did not persist; bootstrapped per HANDOFF.md (env,
`uv sync --all-groups`, dumps restored from the orphan-branch `repro-dumps-4b` tarball).
Sanity gates passed before any edit: `build_supernodes.py --size 4b` re-emitted the 13
review pages, and `--from-exports` on the committed exports reproduced
`multilingual_4b.json` **byte-for-byte** (clean tree). HANDOFF.md deleted (consumed);
this entry + the repo are the record.

### What shipped

- **`multilingual.ipynb` is v2**: 25 cells (was 34), constrained patching ONLY, every
  intervention/readout set from `supernodes/multilingual_4b.json`; graphs load from the
  `supernode_inputs` dumps (tokenization re-derived and length-asserted against the
  manifest); zero in-notebook selection. Chat + raw run side by side inside each
  experiment cell (operation/operand: 2 formats x 3 languages per cell); the language
  swap is raw-only by construction (quote supernodes exist only on the raw pages).
  Executed end-to-end on an A100-80GB, 13/13 code cells, zero errors.
- **`multilingual_helper` v2 API**: `load_supernodes` (approved gate + GLOBAL
  (layer, feature) disjointness — stricter than the addition loader's per-graph key),
  `supernode_suppress_ivs` (per-member recipient positions; documented fallback),
  `supernode_inject_ivs` (value = mult x stored donor-graph act; members without a
  stored act on that graph sit out), `swap_ivs_fn` (paper ramp; endpoint pair hit
  exactly at s = don_max), `choose_swap_end_layer`, `supernode_swap_sweep` (every step
  constrained at the fixed swept ell), `supernode_readout` (per-member positions;
  ref = recipient baseline or stored act with max-act fallback; rows <= ell pinned),
  `readout_layers_above`, `print_readout_row`. Deleted: position_supernode, run_graft,
  the early/late detection scans, swap_interventions, paper_swap(+sweep),
  run_swap_sweeps, the pre-paper top-k overlap variant, feature_label,
  graph_features_at_position. Tests updated in kind (incl. the global-disjointness
  refusal and the constrained-sweep plumbing).
- **`load_transcoder` disk-cache bug found and fixed** (`d35f6e9`): the first-load path
  never forwarded `dtype` to `cache_transcoder`, so the notebooks' bf16 loads wrote
  circuit-tracer's fp32 default — 121 GB for the 4b alone (bf16: 57 GB); the 4b+8b pair
  filled the 369 GB studio disk mid-§H and killed the first execution. Fixed (dtype
  forwarded; regression test), caches rebuilt bf16 (4b 57 GB + 8b 91 GB), hub blobs
  dropped after conversion. The rerun reproduced the aborted run's numbers exactly.

### Executed verdicts (full table in the notebook Summary)

- **Behavior**: raw large/grand/大, chat large/Grand/大; synonym tiny / pet(echo) /
  小(echo); hot-antonym cold/f/冷. `behavior.json` byte-identical to v1's.
- **Shared core**: 107 features in all three chat pruned graphs (of 437–621), 67 raw
  (of 239–439); 11/20 members of the cross-language supernodes sit on all three chat
  antonym pages (6/20 raw) vs the paper's 10/27-in-all-pruned-graphs comparator.
- **Operation swap: not reproduced under the paper's protocol at this selection** —
  p(expected) <= 0.003 in all six jobs, no crossover, baseline top-1 at the ±5x/6x
  endpoint (chat zh 大 @ 1.0), no over-drive degeneration. Cause measured: the reviewed
  antonym supernode's L34 member forces ell in {34, 35}, and the paper-faithful donors
  inject at the mid-sequence operation-word span — unreachable to the final logit
  through <= 1 recomputed layer with frozen attention patterns. All Fig B3 readout rows
  <= L33 => pinned.
- **Operand swap: same protocol-null** — p(expected) = 0.000 everywhere, ell in
  [32, 35]; all say-cold/say-large rows pinned; the single readable row (raw zh
  `opposite (zh)` L34) reads exactly 100% (upstream preserved).
- **Language swap: works in 2/3 directions — reversing v1's null.** en→zh 大 top-1 @
  0.636 (ell=35, crossover 5.5x); fr→en turns English — big @ 0.753 + great @ 0.19,
  the paper's exact FR→EN token (the expected-token metric tracked the model's own
  raw-EN answer `large`, so p_expected under-reports the flip); zh→fr no flip (p_exp
  ~ 0 at every ell). Readouts: all pinned except fr→en's L33 rows — say large (fr)
  **0.0%**, say large (en) **121.6%** of stored: the paper's old-suppressed /
  new-recruited signature on the only readable rows. The reviewed quote features live
  at L23–L34 (Qwen3's raw graphs keep NO early quote-position nodes), so the causal
  handle is LATE — v1's null had steered early-layer (L4–11) activation-scan features
  that carry nothing.
- **Overlap**: en-fr 0.107 > en-zh 0.089 > fr-zh 0.077 mid-third (byte-identical to
  v1). **Default language**: zh 0.766 > en 0.664 >> fr 0.309 (4 shared say-big
  features). **Scale**: 8b > 4b on every pair — 0.132/0.098/0.082 vs 0.107/0.089/0.077
  (en-fr gains most, +23%).
- **The protocol x selection interaction is now measured on a hand-reviewed
  selection**: ell floors 34 / 32 / 32–34 for operation / operand / language, so
  Fig B3/B4 node annotations are unmeasurable here (every row <= ell) while B5's
  measurable rows match the paper. It is a property of where Qwen3's defensible
  supernodes live, not of automated selection.

### Housekeeping

- 23 stale propagate-era artifacts git-rm'd (propagate twins, the raw_* split files now
  keyed inside the constrained JSONs, graph_answers.json, both detection-supernode
  JSONs). New: language_readouts_constrained.json,
  language_ladder_readouts_constrained.json.
- notebooks/README.md: v2 pipeline paragraph covers both notebooks; artifact-naming
  rewritten (constrained-only; `{chat,raw}_{lg}` job keys); the label-cache download
  note dropped (dumps make it unnecessary).
- addition_helper: ruff 0.15 SIM300 autofixes (grid-mask comparisons reordered).

### NEXT (user-stated intent, unchanged)

1. Revisit ADDITION supernodes via the same export-review workflow.
2. Consider raw-primary vs chat-primary framing (paper is raw-only; raw synonym
   caveat: FR echoes pet(it), ZH 小/微 near-tie).
