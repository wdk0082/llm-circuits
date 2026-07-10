# Verification recheck: the notebook reproductions vs the paper, re-audited from sources (2026-07-10)

Branch `verify/paper-recheck`. Independent recheck of `notebooks/addition.ipynb` and
`notebooks/multilingual.ipynb` (+ their helpers) against the biology paper
([biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html))
and the methods paper's addition supplement
([methods.html#graphs-addition](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#graphs-addition)),
**without trusting `notes/biology_digest.md` or the notebooks' own protocol notes**:
both papers' HTML was re-downloaded and re-extracted, every figure SVG was re-fetched
(the Figma exports carry the intervention multiples and outcome percentages as vector
text), rendered, and compared panel-by-panel against the notebooks' committed output
images; every helper function on the causal path (grids, taxonomy labels, steering,
swaps, supernode selection, overlap IOU) was read against the paper's protocol.

## 1. Digest accuracy (notes/biology_digest.md)

Every quantitative claim checked matches the papers: the polymer intervention numbers
(995 @ 98.6% → 997 @ 54.8% under −2×; swap −1×/+1× → 998 @ 66.6%), swap endpoint
strengths (operation −5×/+6×, operand −0.5×/+1.5×, language −5×/+6×), the ≈4× crossover
prose, 20/27 and 10/27 supernode counts, the operand-plot taxonomy definitions, the
introspection dialogue text, suppress-`_6`→98 / suppress-`_9`→91, the parallelogram
argument, the 2,931-feature count (18L model), the overlap protocol
(active-anywhere sets, unrelated-pair baseline), and Fig B5's node-annotation ranges.

**One omission, now fixed in the digest:** the methods paper's supernode-inhibition
figure (`patching-arithmetic-svg`) is titled "Effect of inhibiting a supernode
(**−2×**) on others", while its prose says "perturb it to the **negative of its
original value**" (−1×) — a paper-internal inconsistency the digest's A.4 table
silently sidestepped by listing those interventions with no strength. The
smear-steering figure is `arithmetic-logit-perturb-svg` (no multiple in prose either).

## 2. Findings

Ordered by impact. "m" is this repo's additive-delta convention
(`a_new = (1+m)·a_clean`): m=−1 ablation, m=−2 sign-flip = paper −1×, m=−3 = paper −2×.

### F1 — Input-suppression verdicts cited the wrong strength (fixed)

The addition Summary's suppress-`_6` / suppress-`_9` / inhibit-magnitude rows quoted the
**m=−1 (ablation)** numbers as the reproduction, and the magnitude row (and the DEVLOG
second-session entry) called m=−1 "matched strength". Per §1 the paper's protocol is
m=−2 (prose) or m=−3 (figure) — ablation is strictly weaker than either. This matters
because the phenomenology does not survive the paper's strength on Qwen3-4B:

| Experiment | m=−1 (ablation; previously quoted) | m=−2 (paper prose −1×) |
|---|---|---|
| suppress `_6` (paper: 98, the 9+9 story) | `8` @ 0.29 — the paper's phenomenology | `2` @ 0.99 — no digit story |
| suppress `_9` (paper: 91, no numerology) | `5` weakens to 0.52 | `3` @ 0.96 (also non-numerological) |
| inhibit magnitude (paper: ones path intact) | `5` @ 0.998, ones-path 85–116%, magnitude features 54–66% — clean dissociation | ones answer flips to `3` @ 0.54, sum features 0–23% — dissociation broken |

Disposition: Summary rows re-framed as **reproduced at reduced strength** (the paper's
qualitative claims appear one notch weaker than Haiku's protocol; at paper strength the
4B is over-driven — consistent with the multilingual grafts degenerating at full ±5–6×).
The m=−3 (figure-strength) points were not in the committed run; measured in §4 below.

### F2 — The polymer "−2× suppression" was not the paper's experiment (caveat added)

Fig A5 panel 2 suppresses the `_6+_9` **lookup features only** and *reads out* the
sum=_95/say-995 nodes dropping to 0% — the paper's point is the **weak direct / strong
indirect effect through the sum stage**. The notebook cell steers all 7 active features
(3 lookups **and** 4 sums) together, which destroys the year trivially and tests nothing
about indirectness. The faithful lookups-only run with sum-feature readout: §4.
(The lookup **swap** cell, by contrast, is protocol-exact: flip −1× ⇒ m=−2 plus donors
at +1× of donor-prompt activation, matching Fig A5 panel 3.)

### F3 — "Add-function class reproduced" mislabeled the paper's taxonomy (fixed)

The paper defines **Add Function** features by *operand-conditioned stripe bars*
("one addend satisfies some condition", `add _9` / `add ~57`), and has a separate
**Mostly Active** class ("active on the = token of most of the 10,000 prompts").
The notebook's operator-token probe found 15/16 features grid as **operand-uniform**
(concentration ≈ 1.0) — that is the mostly-active signature, not add-function. The
found result is real and worth keeping; the row now says so, and the stripe-like
add-function hunt (one-operand conditions read at the answer position) is recorded as
a scope gap.

### F4 — `input_mag` supernode pollution (caveat added)

`sel["input_mag"]` = the `magnitude-diag` + **`mixed`** grid classes of the input
panels. The `mixed` bucket is a catch-all: alongside genuine `~46` bands
(`L4f148151`), it contains an **always-on flat feature** (`L0f116505`, activation ≈2.6
on every one of the 10,000 prompts — visible as a solid panel in `grids_input_a.png`;
it was also exactly the feature the earlier duplicate-dedup fix caught, since it peaks
at both operands' positions), plus broadband ripple features (`L0f153246`, `L5f60991`)
and an a≈b-diagonal feature (`L7f31103`). Sign-flipping an always-on feature injects a
large global bias delta, so the m=−2 breakdown in F1 is partly attributable to the
supernode being less surgical than the paper's two clean `~30`/`~59` bands. Follow-up
(queued): re-select `input_mag` with an actual band criterion and re-run.

### F5 — "(6,9) lookups" overstated pair-purity (fixed)

The suppressed lookup-class features auto-label as top-residue pairs
(3,9)/(5,9)/(9,9)/(6,6)/(3,3) — none literally "(6,9)". The labels are **marginal
argmaxes**; Qwen3's lookup features are multi-pair mod-10 lattices whose receptive
fields *include* (6,9) (they must — they are active on the 46+49 prompt, and the
(9,9)-donor swap flips the answer exactly as the paper predicts). This is itself a
model nuance worth stating: Haiku's `_6+_9` feature reads as a single-pair lookup;
Qwen3-4B's equivalents are coarser multi-pair lattices. Wording fixed in the Summary.

### F6 — The intermediate-computation hunt is blind to the paper's mechanism (caveat added)

The `(4+5)*3` cell hunts features with '9' among the label's **top output logits**. The
paper's "computed 9, intermediate step" feature is defined by its *strongest negative*
direct output effect on "9" — a suppressive feature this scan cannot find by
construction. The "partial" verdict stands (the paper hedged its own mechanism), but
the instrument bias is now stated in the row.

### F7 — Stale numbers in DEVLOG (annotated)

- First-session table's suppress-`_6` "`8` @ 0.36" was the retired suite's number; the
  committed notebook reads `8` @ 0.285 (same phenomenology).
- First-session multilingual table's default-language means (zh 0.412 / en 0.338 /
  fr 0.143) are the pre-consolidation notebook's; the committed notebook reads
  **zh 0.766 / en 0.664 / fr 0.309** (same ordering, same conclusion).

### F8 — Notes, no action needed

- The overlap analysis wraps paragraphs in the chat template; the identical template
  tokens inflate the *raw* IOU in all pairs symmetrically and cancel in the paper's
  baseline subtraction (which is what the notebook quotes). Corpus is 28 parallel
  paragraphs — smaller than the paper's (unspecified) set; shape/ordering conclusions
  are robust to this, absolute values less so.
- Operation-swap supernodes are influence-top-10 at the final position with **no layer
  restriction** (the paper picks mid-layer function-vector clusters); since the sweep
  outcome matched and upstream/downstream readouts were not part of our claim, noted
  only. Fig B3–B5's supernode %-readouts (say-large-X preserved/suppressed) were not
  reproduced as readouts anywhere — worth adding to the language-swap null in
  particular (does say-large-zh move *at all* under en→zh?).
- Multilingual crossovers are early and language-dependent (1–3×) vs the paper's
  "fairly consistent ≈4×" — now stated in the Summary row (the headline operation-swap
  verdict is unaffected).

## 3. Verdicts that survive scrutiny (checked, not just re-read)

- **Language swap "differs"** — the central not-reproduced claim — survives: supernode
  construction (early-layer candidate pool, final-token, language-unique, 9–12
  features/language at L4–11), donor sourcing, −5×/+6× endpoints, the proportional
  sweep, and the raw open-quote format control all faithfully mirror Fig B5's protocol
  (whose swapped nodes are the quote-language supernodes; the say-large percentages in
  that figure are readouts, not interventions). p(target-language)=0.000 at every
  strength in every direction, in both formats.
- **Smear "differs"**: m=−3 matches the figure-annotated −2×; the digit-width metric is
  the right ones-circuit analogue of the paper's "range of 5" (the mod-100 `sum=_95`
  target has no digit-split equivalent — note added).
- **Lookup swap** (protocol-exact, margin matches: `8` @ 0.715 vs 998 @ 66.6%),
  **operand swap** (cold/f[roid]/冷, incl. the paper's own lowercase-f outcome),
  **overlap + baseline** (inverted-U, en-fr > en-zh > fr-zh), **scale direction**
  (same-recipe 8b > 4b on every pair), **polymer completion/reuse** (position-resolved),
  **introspection** (6/12 sampled explanations narrate the carry algorithm),
  **corpus reuse** (dates/citations/prices), **two-pathway graphs**, and the operand-plot
  taxonomy geometry (anti-diagonal mod-10 sum stripes, point-lattice lookups, `~46`
  bands, exact-value crosses) — all verified against the paper text and, where the
  paper's figures render (its heatmap rasters are partially blocked server-side, but
  all numbers/labels are vector text), against the figures.

## 4. Recheck measurements (A100-40GB, this session)

Selections reconstructed feature-by-feature from the committed notebook outputs
(cell 8 selection sizes, cell 11 readout keys, cell 9 panel labels); script:
scratchpad `recheck_measurements.py`. m-mode steering needs no stored activations, so
the runs are exact re-instantiations up to bf16/hardware noise (parity rows confirm).

**Parity is exact.** Every measurement that overlaps the committed notebook reproduces
to the printed precision on this different A100 variant (suppress-`_6` m=−1 `8` @ 0.285;
m=−2 `2` @ 0.9858; suppress-`_9` m=−1 0.5161 / m=−2 `3` @ 0.9607; magnitude m=−1
`5` @ 0.9982 / m=−2 `3` @ 0.5368; first-digit readouts 0.395/0.214; polymer all-7
`1` @ 0.7467; calc lookup steer `1` @ 0.8841, width 1.00→1.26) — so the selections were
reconstructed correctly and the new points below are directly comparable.

### 4.1 Input suppressions across the strength ladder (ones answer; baseline `5` @ 0.999)

| Experiment | m=−1 (ablation; previously quoted) | m=−2 (paper prose −1×) | m=−3 (paper figure −2×) |
|---|---|---|---|
| suppress `_6` (paper: 98) | **`8`** @ 0.29 — the 9+9 story | `2` @ 0.99 | `2` @ 0.99 |
| suppress `_9` (paper: 91) | `5` @ 0.52 | `3` @ 0.96 | junk (`,` 0.39, `""` 0.35) |
| inhibit magnitude (paper: ones path intact) | `5` @ 0.998, ones-path 85–116% | `3` @ 0.54, sums 0–23% | junk (flat; top `""` 0.08) |
| … first-digit (low-precision) readout | `9` 0.99→0.40, magnitude features 54–66% | `1` @ 0.21, magnitude features 0% | flat (`1` @ 0.09) |

**Reading:** F1 confirmed and sharpened — the paper's qualitative phenomenology (9+9
numerology; magnitude/ones dissociation) lives at *ablation* strength on Qwen3-4B, and
the model progressively shatters at the paper's −1× → −2×. Same over-drive pattern as
the multilingual grafts at ±5–6×. The Summary's "reproduced at reduced strength"
framing is the measured truth.

### 4.2 The paper-faithful lookups-only suppressions (m=−3 = figure −2×)

| Run | Output | Sum-feature readout |
|---|---|---|
| polymer ones moment (baseline `5` @ 0.982), suppress the **3 active lookups only** | → **`1`** @ 0.77 — 199⟦1⟧, a magnitude-consistent nearby year (cf. the paper's 997 @ 54.8%) | **all 4 active sum features → 0%** (the paper's number) |
| parity: the committed all-7 (lookups+sums) variant | → `1` @ 0.7467 ✓ | — |
| calc ones prompt, suppress the 5 lookup-class features only | → `1` @ 0.88, width 1.26 (the committed smear row) | **all 4 sum features → 0%** |

**Reading:** F2 resolved in the paper's favor — Fig A5 panel 2's weak-direct /
strong-indirect claim **reproduces**: killing only the lookup stage silences the sum
stage completely (0%) and changes the prediction, and the sums add almost nothing
beyond the lookups (0.771 vs 0.747). The polymer-suppression verdict is upgraded from
"reproduced (analog)" to **reproduced**, and the smear "differs" row gains a sharper
mechanistic statement: the flip to `1` happens with the sum stage fully silenced.

Raw numbers: `artifacts/paper_addition/4b/recheck_results.json` (node-local, gitignored).
Follow-ups: five items fully specced (with acceptance criteria and reference numbers)
in the DEVLOG section **"HANDOFF — next A100-80GB session"** — strength ladder,
band-criterion `input_mag`, add-function stripe hunt, negative-weight computed-9
screen, say-large-X readouts for the language-swap null. No code written yet by
decision; the specs are implementation-ready.

## 5. Files changed on this branch

- `notebooks/addition.ipynb` — markdown-only edits (cells 6, 10, 12, 14, 23): strength
  provenance + corrected verdict rows (F1), polymer protocol caveat + recheck resolution
  (F2), taxonomy relabel (F3), `input_mag` caveat (F4), lookup-label wording (F5),
  intermediate-hunt caveat (F6). Code and outputs untouched.
- `notebooks/multilingual.ipynb` — markdown-only edit (cell 24): crossover-consistency
  nuance + the paper's own ZH-echo evidence.
- `notes/biology_digest.md` — the −2× figure annotation + the two methods-figure slugs.
- `DEVLOG.md` — inline corrections at the two mislabeled strength claims + the recheck
  section (mirrors this report).
