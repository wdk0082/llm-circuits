# DEVLOG

> **⏩ CURRENT STATE** (2026-07-09, end of the A100 session): the paper-exact suites
> have been **executed on a local A100-80GB** (not TPU/Slurm) and compared against the
> paper — see "A100 session: suites executed + per-experiment verdicts" below.
> Four selection/protocol bugs were found by the first runs and fixed; two addendum
> scripts (`examples/paper_addition_polymer_probe.py`,
> `examples/paper_addition_swap_addendum.py`) complete the paper's intervention table.
> Artifacts (JSON/PNG/HTML, gitignored) live on the A100 node under
> `artifacts/paper_{addition,multilingual}/<size>/`.

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
