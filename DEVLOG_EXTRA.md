# DEVLOG_EXTRA — final pre-report error scan (2026-07-10)

Independent, adversarial scan of the implementation (`src/llm_circuits/`) and the
reproductions (`notebooks/`) before report writing. Method: line-by-line read of the
four core circuit files; five parallel deep reviews (addition notebook+helper,
multilingual notebook+helper incl. the uncommitted diff, package periphery,
verification/tests/CI/docs, adversarial core-math with ~700 synthetic
circuit-tracer-reference comparisons); **every load-bearing finding below was then
independently re-verified against the code, the real tokenizer, or the committed
artifacts before being recorded.** Per the scan's ground rule, **no code was changed**
— this file only records findings. References: methods paper / biology paper /
circuit-tracer v0.3.1, via the audited `notes/{methods,biology}_digest.md`.

**Tree state at scan time** (branch `handoff/a100-80gb-queue`): uncommitted =
`CLAUDE.md` (perf-note update), `notebooks/multilingual.ipynb` +
`multilingual_helper.py` (Fig B3/B4/B5 %-readout machinery), `notebooks/addition.ipynb`
(`lazy_decoder=False` switch, source-only — its outputs still predate the switch,
harmless since lazy/eager is speed-only), `report/` (new). Note: the multilingual
readout cells were **executed end-to-end mid-scan** (clean top-to-bottom run,
sequential execution counts, artifacts 20:04–20:17, saved 20:20:58, Summary refreshed)
and reproduced every previously committed number bit-for-bit — findings §2 below are
against that final executed state. CI trio re-run on this tree: `ruff check` clean,
`ruff format --check` clean, `pytest` 98 passed / 1 skipped (torch_xla, by design).

Severity: **HIGH** = invalidates or silently changes a reported experiment;
**MED** = wrong-but-contained, or a claim its own data contradicts; **LOW** =
doc/edge-case/hygiene.

---

## 0. Executive summary

| # | Sev | Where | Finding | Status (2026-07-10, same session — commits `9d61004`/`aa7db37`/`3d2e943`; details in DEVLOG third-session addendum 3) |
|---|-----|-------|---------|---------|
| 1.1 | **HIGH** | `addition_helper.digit_token_positions` + cells 7/11 | `suppress_9` was steered at the **teacher-forced answer position**, not the operand's ones-digit token; `input_b` pool built on the wrong positions. Ladder row + Summary verdict for suppress-`_9` measure a different intervention than claimed. | ✅ **fixed + re-measured** (`aa7db37`): digits stop at `=`; operand-position ladder `8` @ 0.711 / **`8` @ 0.678 at −1×** / `2` @ 0.347 — survives paper strength; regression-tested |
| 3.1 | **HIGH** | `interventions.py` (propagate mode) | All notebook interventions run `patch_end_layer=None` with **fixed clean-anchored deltas** — a third protocol that matches neither circuit-tracer's unconstrained *clamp* nor the paper's constrained patching for **multi-layer** feature stacks; module header + docstring actively misdescribe this (docstring claims a default that isn't the default). | **DECISION (2026-07-10, verified against biology.html): adopt (c) — constrained patching; EXECUTABLE PLAN FOR THE NEXT SESSION.** The paper states "Our interventions in this paper use the 'constrained patching' technique" — activations prior to a chosen intervention layer are clamped at perturbed values, the real model runs after it; every dive (incl. multilingual) is under this protocol, and the companion paper's recipe for ℓ is to sweep the patching end layer. Our exact analogue is `run_feature_intervention(..., patch_end_layer=ℓ)` (verified ≡ circuit-tracer's constrained mode; note a single-layer steer with ℓ = its own layer is bit-identical to the committed propagate numbers, so only multi-layer/coupled cases can move). **Steps:** (1) plumbing — add a `patch_end_layer` passthrough to `paper_swap`, `paper_swap_sweep`, `run_graft` (multilingual_helper) and `steer_and_report` (addition_helper); extend `tests/test_notebook_helpers.py` accordingly. (2) ℓ policy — per experiment, sweep ℓ from l_max (last steered layer) to n_layers−1 with `sweep_patch_end_layer` at the paper's quoted endpoint strength, pick the paper's way (most-effective end layer on the target-token metric), state the chosen ℓ next to every result; then run strength sweeps/ladders at that fixed ℓ. (3) scope — addition cell 11 ladders (suppress_6, the position-fixed suppress_9, band + loose magnitude incl. lowprec blocks), cell 13 smears, cell 16 polymer suppression + lookup swap; multilingual cells 8/10/12 sweeps, §G raw sweeps (cells 20/22), and the endpoint %-readout blocks (cells 8/10/21/22). (4) readout caveat — under constrained patching every feature at layer ≤ ℓ is pinned, so %-readouts are meaningful only for nodes ABOVE ℓ (the paper's annotations are downstream nodes — consistent); drop or mark pinned rows. (5) presentation — constrained numbers become the headline paper-protocol results (with their ℓ), propagate numbers are kept as the no-pinning robustness variant; refresh both Summaries + a DEVLOG session section wherever the two disagree. Known deltas to expect: lookup swap `8` @ 0.715 propagate vs `1` @ 0.732 at ℓ=25 (the ℓ-sweep may find a landing layer — that is what the recipe is for); polymer flip survives both (0.747 vs 0.353). Ops: eager decoders make this ~1–2 h total (plumbing + ~30 min re-executions + docs); run the CI trio before committing. Doc fixes from `9d61004` stand regardless. |
| 1.2 | MED | addition Summary | First-digit magnitude baseline misquoted as `9` @ 0.99; the run's own artifact says **0.5626** (inflates the "magnitude path degraded" half of the dissociation ~10×). | ✅ **corrected** (Summary + DEVLOG cite `9` @ 0.5626; historical rows carry ⚠️ notes) |
| 1.3 | MED | addition Summary | "(sum feats → 0%)" at suppress-`_9` m=−1 — the cell's own readout shows 2 of 4 sums at **75% / 30%**. | ✅ **moot after the 1.1 re-run** — at the correct position all 4 sums do read 0% from m=−1; overclaim wording removed |
| 1.4 | MED | addition Summary | "deep low-precision suppression (51–64%)" quotes 3 of 8 readouts; the other five hold ≥97% or rise (one hits 174.8% at m=−2). | ✅ **corrected** ("three most-affected 52–64%; five hold ≥96% or rise, one 176%") |
| 2.1 | MED | multilingual Summary + DEVLOG | zh-unique null tally "10/12 stay at exactly 0 at every strength" is internally inconsistent: 10/12 are zero **at baseline**, only **9/12 at every strength** (the named waker is one of the 10); a second baseline-active member dropping 81% at 6× is counted nowhere. | ✅ **corrected** (tally: 9 always-zero + 1 waker + 2 baseline-active, one −81% @ 6×) |
| 2.2 | MED | multilingual Summary (new readouts) | Operation-swap "say-large suppression" readouts mostly read the **steered source supernode itself** (trio ⊂ source; per-feature values identical in both rows) — not the paper's *downstream* Fig B3 annotations they are compared to; raw-ZH 46.0% (vs paper 20%) folded into "the annotation pattern lands". | ✅ **redesigned; conclusion revised** (`aa7db37`): disjoint `downstream_say_large` added — deep downstream collapse is **FR-only** (1.0% raw @ 6×; EN/ZH keep 69/84%); Summary/DEVLOG updated |
| 2.3 | MED | multilingual Summary + helper docstring | Encoder-side readouts are structurally blind to a feature's **own-layer** decoder delta: sources/donors read propagated response, not the commanded steer (`L0f133356` pinned at exactly 100.0 under a commanded −0.5×; lowest-layer donors at 0% of stored). "Sources suppressed to 10–31%" / "donors recruited to 89–106%" are worded as steering verification; `supernode_readout_pct`'s docstring conflates *injected* with *recruited*. | ✅ **fixed** (docstring + prose: steered rows are propagated response, not steer verification) |
| 3.2 | MED | `attribution_graph.py` ("all" mode + `max_feature_targets`) | Never-attributed features stay in the graph as influence **sinks**; pruned node sets measurably diverge from circuit-tracer's drop-before-prune semantics (34/60 synthetic graphs). Library/serve exposure only — notebooks never set it. | ⬜ open (recorded; library/serve exposure only — notebooks unaffected) |
| 4.1 | MED | `verify_intervention.py` | Prints PASS/FAIL but **always exits 0** (contrast `verify_attribution_graph.py`, which exits 1) — cannot gate a regression. | ✅ **fixed** (`9d61004`): exits 1 on FAIL |
| 4.2 | MED | verification scope | "Numerically equivalent to circuit-tracer" is established for one config (0.6b / fp32 / CPU / one prompt / default all-mode / one constrained m=−2 steer). Steering parity covers neither `value=` donor injections nor propagate mode (which, per 3.1, is where the semantics diverge). | ◻️ acknowledged — cite-with-scope guidance stands; no new cross-verification run |
| 4.3 | MED | tests/CI | The newest verdict-carrying code is untested and outside CI: band-criterion classifier (basis of the magnitude-verdict upgrade; its offline validation is not committed), `direct_token_weights`, `supernode_readout_pct`; `notebooks/` + `verification/` are not in CI's lint/test paths. | ✅ **closed** (`aa7db37`+`9d61004`): 12 helper tests (1.1 regression, band synthetics, readout math, direct-weight vs reference); CI lints `notebooks/` + `verification/` |
| 5.1–5.5 | MED | periphery | `ActivationRecorder.attach` leaks hooks on a bad name mid-list; `compare_models` docstring invites a crashing 1-D input (and silently drops batch>1); feature-label reader's gzip `find()` can misfire on ~1/65k blobs and then crash uncaught; serve steer/reprune not serialized against build/load; serve raw mode uses `n_bos=0` with no sink token (violates the repo's own discipline; notebooks unaffected). | ⬜ open (recorded; periphery, not report-blocking) |

Everything else checked out — see the "verified clean" notes per section; §7 lists what
the report can and cannot safely cite.

---

## 1. Addition reproduction (`addition.ipynb` + `addition_helper.py`)

### 1.1 HIGH — `suppress_9` steered at the teacher-forced answer position; `input_b` pool contaminated

`addition_helper.py:461-495` (`digit_token_positions`) buckets **every** digit token
after `+` into `b_pos` — it never stops at `=` — then keeps the trailing
`len(b_str)` entries. On the teacher-forced ones prompt the trailing slice therefore
**keeps the forced first answer digit and drops b's tens digit**. Independently
reproduced with the real Qwen3 tokenizer on the studied pair (`calc: 46+49=` + forced
`9`):

```
tokens:    [0 sink, 1 'calc', 2 ':', 3 ' ', 4 '4', 5 '6', 6 '+', 7 '4', 8 '9', 9 '=', 10 '9']
positions: {'a_digits': [4, 5], 'b_digits': [8, 10], 'eq': [9], 'plus': [6]}   # correct b: [7, 8]
```

The comment ("guards against digits appearing elsewhere in the prompt") guards the
wrong end: extra digits come *after* the operands on teacher-forced prompts. The bug is
systematic for any 2+-digit sum, not pair-specific. Blast radius:

- **Cell 11**: `("suppress_9", sel["input_9"], pos["b_digits"][-1:])` steers at
  position **10** — the predict-ones/answer position — while cell 10's markdown says
  "we suppress at the operand's ones-digit token position" and the paper suppresses the
  `_9` input supernode at the operand token. The whole suppress-`_9` ladder (m=−1
  `5`@0.516 / m=−2 `3`@0.961 / m=−3 junk) and the Summary row "Suppress `_9` … **reproduced**"
  are measurements of a *different* intervention (the position happens to also hold a
  '9' — the forced first answer digit — so the features are active there and the steer
  does something, just not the paper's experiment). The suppress-`_6` arm uses
  `a_digits[-1:]` = position 5 and is **correct**, so the notebook's narrated
  suppress-6-vs-suppress-9 asymmetry is partly a position artifact.
- **Cell 7**: `groups["input_b"] = position_features(graphs["ones"], pos["b_digits"], …)`
  pools early-layer nodes from positions {8, **10**} and excludes all position-7
  (b-tens) nodes. Mitigation: all six selected `input_9` features also have genuine
  position-8 nodes and pass the `b_top==9` grid filter, so the *membership* is
  plausibly right — the executed steering position is what is wrong. (Side effect:
  features with nodes at both positions occupy two of 16 pool slots → only 12 unique
  features in `taxonomy[input_b]`.)
- **Unaffected**: magnitude supernodes (steered with `position=None`), grid probing
  (peak-over-positions), lookup/sum/polymer/swap cells (final-position), first-digit
  graph experiments (no forced digit → `b_digits` correct there).

Not documented anywhere (DEVLOG's earlier "probed at the wrong position" item was the
since-fixed grids-probe issue). Fix shape (for later; not applied): stop collecting
digits once `=` is seen (or truncate `b_pos` to positions before `eq_pos[0]`), then
**re-run the suppress-`_9` ladder at position 8** and refresh the Summary row.

### 1.2 MED — Summary misquotes the first-digit baseline as 0.99; artifact says 0.5626

Cell 23 magnitude row: "the first digit weakens progressively (`9` 0.99 → 0.54 / 0.51 /
0.43)". The committed run's `artifacts/paper_addition/4b/interventions.json`
(`inhibit_magnitude_lowprec_m-1`) records `baseline_top = [['9', 0.5626], ['?', 0.207], …]`
— verified directly. True progression: **0.5626 → 0.5405 / 0.5072 / 0.4275** (a ~4%
relative drop at m=−1, not the 45-point collapse "0.99 → 0.54" implies). The 0.99 is
the *ones*-prompt baseline (0.9991) copied onto the first-digit graph. DEVLOG's
third-session tables carry the same wrong baseline ("`9` 0.99→0.395", "0.99 → 0.54 /
0.51 / 0.43"). No cell prints this baseline, so a notebook reader cannot catch it.

### 1.3 MED — "(sum feats → 0%)" at suppress-`_9` m=−1 contradicts the cell's own readout

Cell 23: "m=−1: `5` weakens 0.999→0.516 (sum feats → 0%)". Cell 11's printed m=−1
readout (verified): sums `L34f51125` 0.0, `L33f150857` 0.0, **`L27f77434` 75.0,
`L30f158056` 30.2** — 2 of 4, not all. (Row is moot pending the 1.1 re-run, but the
overclaim pattern matters.) Same pattern, smaller: suppress-`_6` "lookup readout
18–45%" omits `L24f49157` at **0.0**.

### 1.4 MED — "deep low-precision suppression … (51–64% band-class readouts)" is a 3-of-8 selection

Cell 23 (loose catch-all row): printed `inhibit_magnitude_loose` first-digit readout is
51.4 / 63.9 / 57.8 for three features and **121.5 / 110.3 / 109.5 / 104.3 / 97.2** for
the other five; at m=−2 one rises to **174.8%** (unmentioned). "band-class" is also not
these features' class (7 mixed + 1 magnitude-diag). The band-vs-loose comparative point
survives; "deep low-precision suppression" overstates a partial, mixed-sign readout.

### 1.5 LOW

- Summary says answer_pos_pool taxonomy = "15 mixed"; the classifier printed **14 mixed
  + 1 band-b** (the visually-rejected `L24f133804` silently folded into "mixed").
- The uncommitted `lazy_decoder=False` comment ends "…the 8b scale section keeps lazy
  decoders" — that section is in *multilingual*.ipynb, not this notebook.
- `steer_and_report`'s docstring mischaracterizes `position=None` ("positions are NOT
  known here") — per `FeatureIntervention` it means "every non-BOS position", which is
  exactly how cell 11 uses it for the magnitude supernodes.

### Verified clean (addition)

m = M_paper−1 wired correctly everywhere (ladder −1/−2/−3, swap flip m=−2 + donors at
`value=`+1× donor activation, polymer m=−3); cell-8 selection dedup exists and covers
every steered selection; swap donors grid-verified (9,9) with disjoint suppress/donor
sets; polymer ones-moment doctrine correct; band criterion internally consistent
(bright-core >0.7·max, band-before-stripe, thresholds 8/15 as documented, sane
degenerate cases); `direct_token_weights` math correct (vocab-mean demeaning, final-norm
scale folding, fp32); probe-position routing correct *except* 1.1; **every other quoted
number recomputed from the notebook's own outputs/artifacts matches to the printed
digit** (77.7/46.0, pair 46+49, suppress-6 ladder, magnitude `5`@0.999 with 94–103%,
smears 1.26/1.03 with `1`@0.8841/`3`@0.9867, polymer `995.`/`5`@0.9824/all-7 `1`@0.7467,
swap `8`@0.7148, computed-9 screen values, introspection 6/12, corpus snippets).

---

## 2. Multilingual reproduction (`multilingual.ipynb` + `multilingual_helper.py`)

State note: the previously-uncommitted readout cells were executed end-to-end mid-scan
(zero cell errors) and the run reproduced every committed continuity number
bit-for-bit (crossovers 0.625×/1.0×/0.875× operand, 3×/1×/2× chat-op, 3×/1×/1× raw-op;
IoU table; 621/437/544 pruned; 107 shared; direct effects). The readout *machinery* is
mechanically sound — mean-of-ratios verified by hand against per-feature printouts;
strength interpolation `source_mult = 1+(src_max−1)·(s/don_max)` lands exactly on the
sweep's pairs; positions, `readout_layers` coverage, dim handling (unsqueezed
`baseline_features` vs squeezed `ablated_features`) all correct. The findings are
about what the new numbers are *claimed to mean*.

### 2.1 MED — zh-unique null tally miscounted ("10/12 stay at exactly 0" → 9/12)

Summary cell + DEVLOG (third session, item 5) say "10/12 late zh-unique features read
exactly 0.00 at baseline and at every strength; one (`L31f104452`) wakes to ~4; one
weakly-en-active member wobbles" — that is 12 accounted as 10+1+1, but the waker **is
one of the 10 baseline-zero features** (verified from
`raw_language_swap_readouts.json`: `L31f104452` baseline 0.0 → 0.36/4.13/3.64). Correct
accounting: **9 always-zero + 1 waker + 2 baseline-active** — and the second
baseline-active member, `L29f56902`, drops **4.69 → 0.90 at 6×** (an 81% fall) and is
counted nowhere. The qualitative conclusion (the zh say-stage never reaches active
scale 15–70; supernode mean ≤7.6% of stored) is unaffected; the headline count is wrong.

### 2.2 MED — operation-swap "say-large suppression" readouts read the *steered source*, not downstream nodes

The say-big trio is *inside* the steered source supernode for the operation swaps
(top-10 influence at the final position): verified in `operation_readouts.json` — e.g.
chat en@3x per-feature values are identical in the `say_big` and `source_antonym` rows
({`L30f27666`: 0.0, `L31f11436`: **224.4**, `L32f100307`: 0.0} in both). The paper's
Fig B3 "~11/10/13% and 0/0/20%" annotate **downstream** say-large nodes distinct from
the steered supernode. The refreshed Summary nonetheless claims "in the raw format at
the paper's full 6× its annotation pattern lands — … say-large driven to 11.2% (EN) /
0.0% (FR) / 46.0% (ZH) vs the paper's ~11/10/13% and 0/0/20%" — with ZH's non-matching
46% folded into "lands", and e.g. chat-EN's "74.8%" mean arising from per-feature
{0.0, 224.4, 0.0}. The **operand**-swap say-big row is clean (trio not steered there)
and genuinely matches the paper (0.0–8.9% vs ≈0–12%).

### 2.3 MED — encoder-side readouts cannot see a feature's own-layer steer; prose words them as steering verification

`run_feature_intervention` readouts re-encode the perturbed **MLP inputs**; a decoder
delta at a feature's own layer is invisible to that feature's own encoding. So steered
supernodes' lowest-layer members read structurally clean values: `source_small`'s
`L0f133356` is pinned at exactly **100.0 in every operand row** despite the commanded
−0.5× (verified in `operand_readouts.json`; it drags "sources suppressed to 10–31%"
upward), and donor rows' lowest-layer members read ~0% of stored (chat en@3x: 6 of 10
at 0.0, mean "45.1%"). The Summary's "synonym donors recruited to 89–106% of
donor-prompt level" and the `supernode_readout_pct` docstring's "the paper's convention
for *injected/recruited* supernodes" conflate injected with recruited — Fig B5's
76–105% is for *downstream recruited* nodes. Clean uses: cell 21's `zh_say`/`en_say`
and cell 10's `say_cold`. The source/donor rows are legitimate propagated-response
measurements but cannot verify the commanded steer; nothing in the notebook says so.

### 2.4 LOW

- `upstream_operand_pct_baseline` in the operation-swap readouts is causally guaranteed
  ≈100% (steer at the final position + frozen attention patterns ⇒ earlier positions
  bit-identical); the 99.3–100.9% spread is cross-pipeline numeric noise. Fine to
  quote (the paper's upstream-100% is equally trivial) but it should carry no
  evidential weight.
- `say_cold` (top-10 of the EN hot graph) contains task-general operation features
  already active on the recipients (`L35f23398`, `L35f125398` appear in both the
  `say_cold` and `upstream_operation` rows), so "recruited to 110/148/65%" partially
  reflects pre-active shared features. (The Summary acknowledges the analogous
  `upstream_operation` mixing, not this one.)
- Prose generosity in §G: "trio holds ≈100% at 1–3×" vs measured means 90.4/84.7
  (per-feature `L31f11436` 70.6 at 3×); "≈half drop to ~50% at 6×" for en-say is
  4/12@47–55 + 4/12@66–82 + 4/12@90–105 (mean 74.5%).
- Latent hazards that do not fire with current data: `operand_token_pos` returning
  None would silently broadcast the steer to every non-BOS position while yielding
  empty readout supernodes; a crossover of exactly 0.0 would make the readout rerun
  ablate donors; cell 22 depends on names defined in cells 8/21 (NameError only under
  a standalone §G re-run); the overlap corpus is chat-wrapped (scaffold-token IoU
  inflation — cancelled by the baseline subtraction actually quoted).

### Verified clean (multilingual)

`paper_swap` (donor∩source dedup — no duplicate steers; single m-conversion; absolute
`value=` donors), sweep/crossover logic (s=0 true baseline; first `p_e>p_b`),
`overlap_curves_paper` (active-anywhere sets, `(i+1)%n` unrelated baseline, mid-third
mean, set-size logging), supernode builders' layer restrictions, `direct_logit_effect`
(fp32, vocab demeaning through final-norm scale), `tokenize_raw` (sink prepend,
N_BOS=1 downstream), `graph_features_at_position` (influence-sorted, absolute token
index); **every DEVLOG/Summary reference number matches outputs digit-for-digit**,
including the new readout claims (the issues above are semantics, not transcription);
all paper values quoted in the new markdown match `notes/biology_digest.md` §B.

---

## 3. Core implementation (`src/llm_circuits/circuits/`)

### 3.1 HIGH — propagate-mode interventions are a third protocol (clean-anchored fixed deltas), and the docs misdescribe it

Facts (each independently verified):

- **No notebook passes `patch_end_layer`** — `grep patch_end_layer notebooks/*.py` is
  empty; every reproduction intervention (all swaps, ladders, polymer, smears) runs the
  default `patch_end_layer=None` = fully-propagating mode.
- In that mode `_steer_base_model` (`interventions.py:196-208`) computes **all** deltas
  once from the *clean* activations: `Δ = (target(clean) − clean)·W_dec`, applied as
  fixed hooks during one real forward (attention patterns frozen, MLPs recompute).
- circuit-tracer's unconstrained mode is **iterative clamping**: `calculate_delta_hook`
  re-anchors on the *perturbed* pass's live activations (`value − current`), so a
  steered feature stays clamped at `value` as its input changes
  (`notes/methods_digest.md` §5.2, verbatim from the installed source). The biology
  paper's stated protocol is **constrained patching up to a chosen intervention layer**
  (`notes/biology_digest.md:9,212`) — in our API, `patch_end_layer=l_max`-style, which
  pins MLP outputs *between* steered layers to clean.

Consequence: for interventions spanning **≥2 causally-coupled layers** — exactly the
notebooks' flagship cases (say-big trio L30/31/32 inside swap sources, multi-layer
donor lists, L24/L25 lookup sets, L27–L34 sum sets) — a steered feature downstream of
another steer receives `(value − clean)·W_dec` instead of circuit-tracer's
`(value − current)·W_dec` (discrepancy `(current − clean)·W_dec` per feature; e.g. an
m=−1 "ablation" of a coupled stack leaves knock-on residuals where circuit-tracer's
clamp silences exactly). Single-layer or uncoupled interventions are **exact** (first
steered layer sees an unperturbed input, so clean == current), and the verified
constrained mode is unaffected (circuit-tracer uses the clean basis there too).

This does not make the notebook experiments *invalid* — clean-anchored delta steering
with real propagation and frozen patterns is a coherent, defensible protocol — but
(a) it is **not** what the module claims ("Faithful to circuit-tracer's
`feature_intervention`", `interventions.py:1`, unqualified — DEVLOG task-4 correctly
scopes its equivalence claim to *constrained* interventions), (b) it is not the
paper's stated constrained-patching protocol either, and (c) the divergence's size on
the reported numbers is unquantified. Compounding doc bug: the
`run_feature_intervention` docstring says "**Defaults to the last steered layer** (max
downstream recompute)" (`interventions.py:349-351`) while the signature default is
`None` (= propagate) — a reader following the doc believes they get constrained-to-
`l_max` by omitting the argument. Suggested resolution (not applied): either re-anchor
propagate-mode deltas iteratively (clamp semantics), or document propagate mode as a
deliberate protocol choice, fix the default sentence, narrow the header claim — and
optionally quantify by re-running one swap with `patch_end_layer=l_max` for comparison.

### 3.2 MED — `"all"` mode + `max_feature_targets` leaves un-attributed features as influence sinks

`attribution_graph.py:756-762` caps *targets* but keeps never-attributed features as
source-only nodes; circuit-tracer drops unselected features from the adjacency before
pruning. The sinks absorb propagated influence and skew row-normalisation: pruned node
sets differ from drop-first semantics in **34/60** synthetic graphs. Exposure:
library/serve API only (`BuildRequest.max_feature_targets`; notebooks never set it —
they use `influence_ranked` + `max_feature_nodes`, which is exact, see below). The
inline comment acknowledges only the top-by-activation selection bias, not the sink
effect.

### 3.3 Smaller items

- **MED-LOW** `feature_selection` never validated (`attribution_graph.py`): any typo
  silently means `"all"` *and* flips `max_feature_nodes` from an influence target-cap
  into an activation pre-cap. `update_interval=0` would silently attribute nothing.
  Also `max_feature_targets` is silently ignored in `influence_ranked` mode (serve
  passes it in both modes).
- **MED-LOW** `freeze_attention=False` + `patch_end_layer=None`: baseline logits come
  from the forced-eager capture pass, the intervened pass runs SDPA → every delta
  carries an eager-vs-SDPA noise floor (≈1e-2 logits at bf16) and m=0 is not exactly
  zero. Notebooks always freeze (default True); serve UI exposes the toggle.
- **LOW** `_warn_duplicate_interventions`: `position=-1` + `position=seq-1` alias the
  same delta row (2× accumulation, verified) but are not warned; broadcast
  (`position=None`, which skips BOS) + explicit `position=0` is warned though there is
  no physical overlap.
- **LOW** hand-loaded JSON graphs only: cyclic/over-deep graphs get silently truncated
  influence where circuit-tracer raises (`max_iter` fall-through); parallel duplicate
  `(target,source)` edges share one summed score. Unreachable for built graphs (edges
  strictly layer-forward; the builder cannot emit parallel edges — both verified).
- **LOW** `edge_threshold=1.0` exactly is an FP boundary (kept-edge sets can differ
  from circuit-tracer in the last ulp; 0 mismatches in 280 runs at every other tested
  threshold pair). Serve's reprune slider permits 1.0.
- **LOW** dead code paths: `LocalReplacementModel(ablations=…, frozen_errors=…)` have
  zero callers; `_apply_ablations` applies `position=None` at BOS (inconsistent with
  the live path, which skips BOS) and BOS ablations are output-inert due to the splice.
- **LOW** cosmetics: `_make_frozen_attn_forward` type hint promises a 3-tuple, returns
  2 (correct for the pinned transformers); `capture_constants` restores only layer-0's
  attn impl (correct only because Qwen3 layers share one config object); stale comment
  "features are already squeezed" in `attribution_graph.py`; serve schema describes
  influence mode as "build full matrix then keep top-N" (engine actually runs the
  dynamic never-materialise selection).

### Verified clean (core)

Adversarial synthetic comparisons against verbatim ports of circuit-tracer v0.3.1:
**influence_ranked selection loop exactly equivalent** (0/280 mismatched visited sets
across cap/batch/update-interval combinations; apparent mismatches were tie-order
artifacts); **full pruning pipeline node- and edge-set identical** on 240+ randomized
graphs at all non-boundary thresholds incl. tie-heavy weights (`_find_threshold` tie
handling identical); `_compute_logit_weights` fallback scale-invariant (pruning
identical); `cap_features_by_influence` exact; constrained-intervention equivalences
(LN-freeze iff full range; pattern-freeze coupling; pins below first steered layer are
no-ops; readout = encode perturbed MLP input, matching circuit-tracer's cache;
sweep/progressive baselines consistent; same-layer deltas superpose); frozen RMSNorm
hook bit-for-bit HF Qwen3 order; BOS error exactly zero (emergent from the splice);
`max_targets_guard` planned-count formulas correct in both modes; all 46 core unit
tests pass.

---

## 4. Verification & CI layer

- **MED** `verify_intervention.py` computes `ok` and prints PASS/FAIL but **never sets
  a failing exit code** (verified; `verify_attribution_graph.py:509` exits 1 properly).
  The DEVLOG "steering PASS" is a recorded manual reading; scripting it cannot catch a
  regression.
- **MED (scope, documented)** The "numerically equivalent to circuit-tracer"
  conclusion (DEVLOG task-4) rests on one config: Qwen3-0.6b, fp32, CPU, one prompt,
  default all-mode attribution, one constrained m=−2 steer. The steering check covers
  neither `value=` donor injections nor propagate mode (§3.1). Internal-exact checks
  (forward exactness, linearity identity, autograd spot-checks) do validate the
  algorithm beyond the instance; the cross-implementation claims should be quoted with
  their scope.
- **MED** Check-design asymmetries (empirically moot on the recorded 11/11 run): the
  explicit missing-edge-mass check is one-directional (ct→ours); `max|dw|` is printed
  but not asserted; pass thresholds (Jaccard ≥0.98/0.9, cos >0.99) are far looser than
  the recorded values (all 1.0000) — a materially degraded re-run could still print
  11/11.
- **MED** Newest verdict-carrying code is untested and outside CI: the band-criterion
  classifier (sole basis of the magnitude-verdict upgrade; its "validated offline
  against run-#1 grids + synthetics" is not committed anywhere), `direct_token_weights`,
  `answer_pos_pool`/`grids_for` routing, `supernode_readout_pct`. CI lints/tests only
  `src tests examples`; `notebooks/` and `verification/` are outside (they do currently
  pass ruff — checked). ci.yml also only triggers on push/PR to `main`, so this branch
  never ran server-side CI.
- **LOW** The recorded steering PASS predates the current `interventions.py` (readout
  capture + duplicate warning added since); verified additive/gated, and
  `git diff a34df49..HEAD -- src/llm_circuits/circuits/` confirms
  attribution/pruning/local-replacement are byte-unchanged since the 11/11 run. A
  re-run is cheap hygiene.
- **LOW** DEVLOG's 11/11 table lists 10 rows (omits the "same logit nodes" check;
  count right, row missing).

---

## 5. Package periphery

- **MED** `instrumentation/hooks.py:73-76`: `ActivationRecorder.attach` registers
  hooks *before* the `try`; a bad module name mid-list permanently leaks the
  already-registered hooks (verified — error-path only; in-repo callers use fixed
  templates).
- **MED** `circuits/replacement_model.py:410-445`: `compare_models` docstring offers
  `(seq,)` input, which crashes in the HF forward; batch>1 silently drops all but the
  first row. In-repo callers pass 2-D batch-1.
- **MED** `transcoders/feature_labels.py:105,150`: the blob format is a fixed 4-byte
  size header + gzip, but `_read_feature_blob` *scans* for the gzip magic
  (`chunk.find(b"\x1f\x8b")`); when the size header itself contains `1f 8b` (~1/65k
  blobs, data-dependent) decompression raises `BadGzipFile`/`EOFError` — **OSError,
  not in the `except (IndexError, ValueError)`** — so label loading crashes instead of
  skipping one label. Positive counterpart (verified empirically on the cached 4b
  repo): the layer/feature **indexing is exact** (offsets fence-posts correct; decoded
  blobs carry the requested index) — no mislabeling risk in any figure.
- **MED** `serve/engine.py`: steer/sweep/reprune are not serialized against
  build/load (`_free()` can null the model under a concurrent forward; label cache
  mutated unlocked) — multi-client exposure only; the bundled frontend masks it.
- **MED** `serve/engine.py:242-244`: raw (non-chat) mode tokenizes with **no prepended
  sink token and `n_bos=0`**, violating the repo's own attention-sink discipline
  (`replacement_model.py:11-14`; the notebooks' raw path prepends and uses
  `n_bos_tokens=1`) — inflated position-0 error node in the UI's raw mode, silently.
  Notebooks unaffected.
- **LOW** serve chat branch hardcodes `n_bos=1` instead of using `prepare_messages`'
  returned count; loader `device=None` docstring wrong (resolves via circuit-tracer's
  default, not settings — diverges on MPS); `load_transcoder`'s auto-cache path
  confirmed fp32 (undocumented at the call site) and `cli.py transcoder-cache` exposes
  no `--dtype`, so the documented bf16 disk-halving trick is Python-only (verified);
  `visualization._make_visible` replaces space with space (no-op — leading-space vs
  bare tokens indistinguishable in labels; inherited by graph_explorer);
  `serve/app.py` steer/sweep/reprune map only `RuntimeError` (a bad `patch_end_layer`
  from the UI's free-text box surfaces as a bare 500); `LLM_CIRCUITS_SERVE_MOCK=0`
  still forces the mock (truthy-string check); `seed_everything` not re-exported as
  CLAUDE.md implies and unused; `replacement_model` KV-cache-generation BOS hazard and
  CLT buffer contamination on mid-forward exceptions (both dormant — no callers).

Verified clean (periphery): feature-label indexing exact; m-convention consistent
end-to-end (schemas → engine → UI tooltip); edge direction/sign colors/layer axes
consistent across visualization, explorer and serve payloads; bf16 caching claims
match the code; `hf_loader` deliberately leaves attn implementation alone (the capture
pass flips to eager and restores — required, verified against transformers 4.57);
chat path passes `enable_thinking=False` everywhere; `examples/demo.py` matches all
current APIs; the circuit-tracer import rule holds (loader + accepted isinstance shim
+ verification only).

---

## 6. Docs / state housekeeping

- `hpc/run_multilingual_notebook.sbatch` `--time=01:30:00` assumes warm caches; a cold
  first run (fp32 auto-cache 113 GB + ~45 GB labels + the 8b set for §H) will likely
  exceed it, and with `--inplace` a wall-clock kill loses the executed notebook.
- DEVLOG's "Repo cleanup" section still describes `examples/` as holding the retired
  demo scripts (`compare_*`, `graph_explorer`, …); the tree has only `demo.py` +
  `tpu_smoke_test.py`. README/CLAUDE.md are correct; only the historical DEVLOG prose
  is stale.
- `notebooks/README.md` wall-times ("≈40/60 min") are the second-session numbers;
  third session measured ~21–25/~35 min (conservative, not wrong).
- CLAUDE.md's HPC/Slurm preamble does not match the Lightning-studio nodes the last
  three sessions actually used.

---

## 7. Implications for the report (what to cite / not to cite)

**Do not cite as-is until fixed/re-run:**
1. The suppress-`_9` ladder row and any suppress-6-vs-9 contrast (§1.1 — wrong steering
   position; needs the helper fix + re-run at position 8).
2. The first-digit magnitude progression "0.99 → …" (§1.2 — baseline is 0.5626).
3. "Sum feats → 0%" at suppress-`_9` m=−1 (§1.3) and "deep low-precision suppression
   51–64%" (§1.4) as written.
4. The operation-swap say-large %-readouts as a match to the paper's Fig B3
   *downstream* annotations (§2.2), and "donors recruited to 89–106% / sources
   suppressed to 10–31%" as *steering verification* (§2.3). The operand-swap say-big
   row (0.0–8.9% vs paper ≈0–12%) **is** clean and citable.
5. "10/12 zh-unique features stay at exactly 0 at every strength" — the correct count
   is 10/12 at baseline, 9/12 at every strength, with `L29f56902` (−81% at 6×)
   accounted (§2.1). The mechanism conclusion itself stands.

**Cite with scope qualifiers:**
6. "Numerically equivalent to circuit-tracer" → scope to the tested configuration
   (0.6b/fp32/CPU, default attribution, constrained steering) (§4).
7. Notebook interventions = clean-anchored, fully-propagating decoder-delta steering
   with frozen attention patterns — a coherent protocol, but not circuit-tracer's
   unconstrained clamp nor the paper's constrained patching for multi-layer coupled
   supernodes; describe it as what it is (§3.1). Single-layer/uncoupled results are
   protocol-exact.

**Safe:** everything else in both Summaries — every other number was re-derived from
the notebooks' own outputs/artifacts during this scan and matched to the printed digit;
graphs, taxonomy, operand swap, polymer suite, overlap+baseline, 4b-vs-8b scale
result, and the language-swap null (with §2.1's corrected tally) all stand.
