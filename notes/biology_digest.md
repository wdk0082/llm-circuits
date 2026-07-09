# Reference digest: "On the Biology of a Large Language Model" — Addition & Multilingual case studies

Source: https://transformer-circuits.pub/2025/attribution-graphs/biology.html (Lindsey, Gurnee, Ameisen, Chen, Pearce, Turner, Olah et al., Anthropic, 2025).
Companion methods paper: https://transformer-circuits.pub/2025/attribution-graphs/methods.html (addition supplement under anchor `#graphs-addition`).
Fetched and verified against the raw HTML on 2026-07-09. Studied model: **Claude 3.5 Haiku** (features from a cross-layer transcoder, "h35" scan). Scaling comparisons use a **smaller 18-layer research model ("18L")**.

**Method conventions needed to read the results below**
- Attribution graphs are computed on a local replacement model (CLT features + error nodes + frozen attention); nodes grouped by hand into "supernodes".
- Interventions use **"constrained patching"**: activations of the patched features are clamped at perturbed values up to a chosen intervention layer, so effects before that layer are suppressed and effects after it are computed by the real model.
- "Suppression"/inhibition = clamping a feature (or supernode) to a **negative multiple of its original activation** (−1× = sign-flip ablation; figures annotate the multiple used, e.g. −2×, −5×). Positive injections are quoted as multiples of the donor prompt's activation (e.g. +6× = six times the activation the features had on the source prompt).
- Figure notation: `_` = "any digit can go here" (e.g. `_6` = ends in 6), `~` = "approximately" (e.g. `~36` = number near 36). Percentages printed next to supernodes are activation relative to baseline (100% = unchanged); percentages next to tokens are output probabilities.
- Static figures are Figma SVG exports at `https://transformer-circuits.pub/2025/attribution-graphs/figma/<slug>.svg`; interactive attribution graphs at `https://transformer-circuits.pub/2025/attribution-graphs/static_js/attribution_graphs/index.html?slug=<slug>`.

Local figure copies: `(session-local, re-download via the URLs below) figures/` (SVGs reference heatmap rasters in the `figures/png/` subdirectory, which were downloaded too, so the SVGs render locally; one large background raster used by the three multilingual "swap" SVGs is AccessDenied on the server itself — all numbers/labels in those figures are vector text and fully preserved).

---

# A. Addition case study (§ "Addition", anchor `#dives-addition`)

Behavior: prompt `calc: 36+59=` → model completes `95`. Biology-paper section builds on the fuller study in the methods paper.

## A.1 Figures

### Fig A1 — Operand plots taxonomy ("heatmap figures")
- Slug: `operand-plots-biology-svg` → local `figures/addition_operand_plots.svg` (+ 3 heatmap PNGs in `figures/png/`).
- URL: https://transformer-circuits.pub/2025/attribution-graphs/figma/operand-plots-biology-svg.svg
- Caption: "Example operand plots for feature types active on addition prompts of the form 'calc: a+b=' for a, b in [0,99]."
- What it plots: **feature activity on the `=` token as a 100×100 heatmap over the two operands** — x-axis "The value of addend b", y-axis "The value of addend a", ticks 0–90; 10,000 prompts `calc: a+b=`, a,b ∈ [0,…,99]. Three example panels: an **"add X" / add-function feature**, a **lookup-table feature** ("a is near 36 and b is near 60"; also "a ends in 9 and b ends in 9"), and a **sum feature** ("a+b is 5 modulo 10", showing anti-diagonal stripes); a "6+9" point is highlighted.
- Geometric taxonomy (paper text, verbatim structure):
  - **Diagonal lines** = features sensitive to the **sum**.
  - **Horizontal / vertical lines** = features sensitive to the **first or second operand** respectively.
  - **Isolated points** = **"lookup table" features** sensitive to **combinations** of inputs.
  - **Repeating patterns** = **modular** information (e.g. "the last digit is X mod 10").
  - **Smeared patterns** = **lower-precision** versions of the above categories.

### Fig A2 — Attribution graph for calc: 36+59=
- Slug: `36-59-svg` → local `figures/addition_36_59_attribution_graph.svg`. Interactive graph slug: `calc-36-plus-59`.
- Caption: "A simplified attribution graph of Haiku adding two-digit numbers. Features of the inputs feed into separable processing pathways."
- Node inventory (bottom → top), with in-figure annotations:
  - **Input features** (on tokens 36 / 59): `36`, `_6`, `~30` and `59`, `_9`, `~59` — "The model has features specific to the ones digit and to the approximate magnitude, at various scales."
  - **Add-function features**: `add _9`, `add ~57` — "The model separately determines the ones digit of the number to be added and its approximate magnitude. Operand plots show vertical or horizontal stripes."
  - **Lookup-table features**: `_6 + _9`, `~36 + ~60`, `~40 + ~50` — "The model has stored information about particular pairs of input properties. They take input from the original addends (via attention) and the Add Function features. Operand plots are points, possibly with repetition (modular) or smearing (low-precision)."
  - **Sum features**: `sum = _95`, `sum = _5`, `5_`, `sum ~92` — "The model has finally computed information about the sum: its value mod 10, mod 100, and its approximate magnitude."
  - Output: `95`. Note: "Most computation takes place on the '=' token".
- Two parallel pathways described in the text: **low-precision path** "add something near 57" → lookup "add something near 36 to something near 60" → "the sum is near 92"; **high-precision modular path** "left operand ends in a 9" → "add something ending exactly with 9" → "add something ending with 6 to something ending with 9" → "the sum ends in 5". "These combine to give the correct sum of 95."

### Fig A3 — Dataset examples of the _6+_9 lookup feature (static PNG)
- URL: https://transformer-circuits.pub/2025/attribution-graphs/png/img_9ac0f261fe832699.png → local `figures/addition_6plus9_feature_dataset_examples.png` (1542×488).
- Shows ~10 text snippets with token-level activations of feature `h35/b12406787` (the `_6+_9` lookup feature) highlighted, all non-arithmetic surface forms: a zoological citation ("Zool. Garten. 16. Jhg. **1**875"), French mortality statistics, Maroc Telecom financial figures, a biography date "1086 H. (**1**675–76)", astronomical minute tables ("228.17.**0**6,53"), "J. de Phys., Vol. 36, p. L-69, **1**975", "Epilepsia, Vol. 36, Suppl. 4, **1**995", a revenue table ("$119,6**00** | $35,820"), and "K. Whang, etc. Polymer, 36, 837, **1**995".
- The paper also inlines three long text examples (astronomical measurements where durations 38–39 min starting at minute 6 predict end minute 45; a customer/revenue table where Cost $35,820 = $26,880 + $8,970; the Polymer citation).

### Fig A4 — Journal-citation attribution graph (Polymer)
- Slug: `polymer-svg` → local `figures/addition_polymer_journal_context.svg`. Interactive graph slug: `polymer-add-9`.
- Caption: "A simplified attribution graph of Haiku completing an academic journal citation. The same lookup table feature that is active on addition prompts helps to infer the correct citation year."
- Prompt: "…(K. Whang, etc. Polymer, 36, 837, 1" → completes "995". Five recognizable arithmetic features from the simple graphs (shown with their operand plots) — `add ~36`, `add _6`, `_6 + _9`, `~36 + ~60`, `sum _95` — combine with **two journal supernodes**: "journals founded in ~1960" and "journals founded in years __0", feeding `say 995` / `say 99_` / "say a recent year".
- Text: the `_6+_9` feature "activates when the journal volume number (36 here) ends in 6 and the year before the founding of the journal ends in 9 (1959 here), such that the year of publication of the volume will end in a 5."

### Fig A5 — Journal-citation intervention experiments
- Slug: `arithmetic-polymer-intervention-svg` → local `figures/addition_polymer_journal_intervention.svg`.
- Caption: "Intervention experiments to establish the causal role of addition lookup table features in academic journal citations." Three panels (prompt "Polymer, 36, 837, 1"):
  1. **Baseline**: all supernodes 100%; top outputs **995 @ 98.6%**, 997 0.1%, 994 0.4%, 993 0.1%, 996 0.1%, 998 0.1%.
  2. **Suppress `_6+_9` at −2×**: `~36+~60`, `say 99_`, "say recent year" stay ≈100% but `sum = _95` and `say 995` drop to **0%**; top outputs become **997 @ 54.8%**, 993 22.2%, 998 8.5%, 994 3.7%, 992 2.3%, 999 1.8% (995 no longer in top outputs).
  3. **Swap lookup feature**: `_6+_9` (features b12406787+b6290003) at −1× plus `_9+_9` (feature b24589983) at +1× → `sum = _98` and `say 998` activate (sum=_95 → 0%): top outputs **998 @ 66.6%**, 995 18.2%, 997 7.2%, 993 2.2%, 994 1.8%, 996 0.7%. Text: "changes the ones digit of the prediction in the expected way (from 1995 to 1998)".
- Text conclusion: "Suppressing the lookup table feature has a weak direct effect on the output prediction, but its indirect effect on the sum and output features is strong enough to modify the model's prediction."

### Fig A6 — Intermediate computation: assert (4 + 5) * 3 ==
- Slug: `addition-intermediate-svg` → local `figures/addition_intermediate_4plus5times3.svg`. Interactive graph slug: `order-of-operations-paren`.
- Caption: "A simplified attribution graph of Haiku computing the answer to an arithmetic expression with two steps. A lookup table feature active on addition prompts is used as an intermediate result for the larger expression."
- Prompt `assert (4 + 5) * 3 ==` → model completes `27`. Nodes: addition lookup `4 + 5 → 9`; multiplication lookup `3 × 9 → 27` plus `multiply by 3` and `say multiple of 9` pathways; `computed 9 or _9_`; **`computed _9 as an intermediate step`**; **expression-type feature "(a + b), then multiply"**; `say 27`.
- In-figure annotations: expression-type feature "Detects that a sum (a+b) will be used as an intermediate result and multiplied by another quantity. Upweights lookup tables for the addition and the multiplication, and a feature that flags the addition as an intermediate step." The 4+5 lookup "Upweights the '9' response directly (incorrectly in this case), as well as features that make use of 9 as an input." Intermediate-step feature "Identifies that 9 is an intermediate step, but not the final answer."

## A.2 Key quantitative results
- Operand plots computed over **10,000 prompts** (`calc: a+b=`, a,b ∈ [0,99]), feature activity read on the `=` token.
- Polymer intervention numbers: see Fig A5 (98.6% → 54.8%-for-997 under −2× suppression; 66.6%-for-998 under _9+_9 swap).
- Introspection dialogue (verbatim, from paper — graph for this longer prompt attributed from "95" showed "the same set of input, add, lookup table and sum features as in the shorter prompt"):
  - Human: "Answer in one word. What is 36+59?" — Assistant: "95"
  - Human: "Briefly, how did you get that?" — Assistant: "**I added the ones (6+9=15), carried the 1, then added the tens (3+5+1=9), resulting in 95.**"
  - Paper's reaction: "Apparently not!"
- Supplementary (companion methods paper, verified in its HTML):
  - Virtual-weight / global-circuit analysis restricted to features **active on ≥10 of the 10,000 addition prompts**; **2,931 features** are "prominent on two-digit addition problems" in the smaller 18L model.
  - Suppressing the `_6` input feature → model outputs **98** ("the tens digit from the original problem is preserved by the other magnitude signals but the ones digit is that which would result from adding 9 to itself"); suppressing `_9` → outputs **91**, "not 92, so such numerology must be taken with a grain of salt".
  - Inhibiting low-precision input features (`~30` and `~59`) suppresses the low-precision lookup-table features, the magnitude sum feature and the appropriate sum features "while leaving the ones-digit pathway alone".
  - Negatively steering the `_6 + _9` lookup features "smears the result out over a range of 5", while negatively steering the final `sum=_95` feature "smears the result out to a wider band (perhaps coming from sum~92 features)".
  - "Parallelogram constraint" argument for why lookup-table features must exist: feature preactivations are affine in inputs, so "if a feature f is active on two inputs of the form x+y and z+w, then it must be active on at least one of the inputs x+w or z+y" — a direct input→sum computation in one nonlinear step is impossible.

## A.3 Qualitative conclusions (precise statements)
1. **Parallel-pathways claim:** Haiku "split the problem into multiple pathways, computing the result at a rough precision in parallel with computing the ones digit of the answer, before recombining these heuristics to get the correct answer." [Expected to generalize as a motif; exact feature inventory is model-specific.]
2. **Lookup-table features:** a key step is performed by features that "translate between properties of the input (like the two numbers being summed ending in 6 and ending in 9) and a property of the output (like ending in 5)"; "Like many people do, the model has memorized the addition table for one-digit numbers." [Motif expected to generalize; architecturally motivated by the parallelogram constraint.]
3. **Not the human algorithm:** "The other parts of its strategy, however, are a bit different than standard algorithms for addition used by humans." No sequential carry procedure is found.
4. **Modular / "ends in 5" features:** sum information is represented redundantly at several moduli and precisions (`sum = _5` i.e. ends-in-5 mod 10, `sum = _95` mod 100, `sum ~92` magnitude; input-side `_6`, `_9` mod-10 features; repeating operand-plot patterns = "the last digit is X mod 10"). [Expected to generalize.]
5. **Introspection mismatch:** the model explains its answer with the standard carry algorithm, which does not match its actual mechanism — "a simple instance of the model having a capability which it does not have 'metacognitive' insight into. The process by which the model learns to give explanations (learning to simulate explanations in its training data) and the process by which it learns to directly do something (the more mysterious result of backpropagation giving rise to these circuits) are different." [Expected to generalize to any chat-tuned model; the specific explanation text will differ.]
6. **Generalization to input contexts:** the same `_6+_9` lookup feature is causally reused wherever "there is often a reason to predict the next token might end in 5, coming from adding 6 and 9" — astronomical tables, financial/tax tables, academic citation years. "For each of these cases, the model must first figure out that addition is appropriate, and what to add; before the addition circuitry operates. Understanding exactly how the model realizes this … is a challenge for future work." [Reuse-of-arithmetic-features motif expected to generalize; the specific contexts found depend on the feature-vis dataset.]
7. **Flexibility of computational role / intermediate results:** on `assert (4 + 5) * 3 ==`, "the '4 + 5' features have two effects with opposite signs — by default they drive an impulse to say '9,' but, in the presence of appropriate contextual cues indicating that there are more steps to the problem …, they also trigger downstream circuits that use 9 as an intermediate step." Expression-type features "are responsible for nudging the model to use some of these circuits in favor of others"; lookup-table features "act as the workhorses of the basic computations".
   - Caveat stated in the paper: the "computed 9 as intermediate step" feature's strongest negative direct output effect suppresses "9", but "this negative influence is rather weak in the attribution graph (the strongest inhibitory inputs to the '9' output are error nodes), so it is unclear if this suppressive mechanism is significant in the underlying model." [Keep this hedge when checking a reproduction.]

## A.4 Interventions (summary list)
| Where | Intervention | Outcome |
|---|---|---|
| Polymer citation (biology paper) | `_6+_9` at −2× | 995 (98.6%) → 997 top (54.8%); sum=_95, say-995 nodes → 0%; other nodes ≈100% |
| Polymer citation | `_6+_9` −1× and `_9+_9` +1× | prediction 1995 → 1998 (998 @ 66.6%); sum=_98/say-998 recruited |
| calc: 36+59= (methods paper) | suppress `_6` | output 98 |
| calc: 36+59= (methods paper) | suppress `_9` | output 91 (not 92 — caution) |
| calc: 36+59= (methods paper) | inhibit `~30`, `~59` | low-precision lookup/magnitude-sum/sum features suppressed; ones-digit path intact |
| calc: 36+59= (methods paper) | negative steer `_6+_9` vs `sum=_95` | smears output over range ~5 vs a wider band |

---

# B. Multilingual circuits (§ "Multilingual Circuits", anchor `#dives-multilingual`)

Behavior — three prompts with identical meaning:
- English: `The opposite of "small" is "` → **big** (figures show top prediction "large"; either English size-antonym counts)
- French: `Le contraire de "petit" est "` → **grand**
- Chinese: `"小"的反义词是"` → **大**

Central claim: "these three prompts are driven by very similar circuits, with shared multilingual components, and an analogous language-specific component" (a combination of language-invariant and language-equivariant circuits).

## B.1 Figures

### Fig B1 — Overview: three parallel simplified graphs
- Slug: `multilingual-overview-hover-1-svg` → local `figures/multilingual_overview_part1.svg`. Interactive graph slugs: `opposite_of_small` (EN), `opposite_of_petit` (FR), `opposite_of_small_zh` (ZH).
- Caption: "Simplified attribution graphs for translated versions of the same prompt, asking Haiku what the opposite of 'small' is in different languages. Significant parts of the computation appear to be overlapping 'multilingual' pathways. Note that these are highly simplified…"
- Per-language supernode skeleton (identical structure): `opposite (lang-specific)` + `small (multilingual)` → `antonym (multilingual)` →(dotted, attention/QK-mediated) `say large (multilingual)`; separately `quote (lang-specific)` → `say large (lang-specific)` → top prediction large / grand / 大. Multilingual nodes are shared/aligned across the three panels.

### Fig B2 — Overview of the three interventions
- Slug: `multilingual-overview-2-svg` → local `figures/multilingual_overview_part2.svg`.
- Caption: "Overview of the three kinds of intervention experiments we'll perform, intervening on the operation, the operand, and the language."
- Three panels on the **English** prompt: **Operation swap** (antonym → synonym) → prediction "little"; **Operand swap** (small → hot) → "cold"; **Language swap** (English → Chinese) → 大 (zh: big).

### Fig B3 — Operation swap in all three languages
- Slug: `multilingual-swap-operator-svg` → local `figures/multilingual_swap_operation_antonym_to_synonym.svg`.
- Caption: "Interventions on the operation, swapping antonym for synonym features in three different language input cases."
- Protocol per language: **antonym supernode −5×; synonym supernode +6×** (synonym features taken from the *English* prompt `A synonym of "small" is "`). Line charts: next-token probability (0–1) vs intervention strength 0×–6× (EN tokens: little, tiny, large, big; FR: min[uscule], petit, grand; ZH: 微, 小, 矮, 大).
- Outcomes: EN large → **little**; FR grand → **min[uscule]** (fr: tiny); ZH 大 → **微** (zh: tiny). Node annotations: upstream nodes (opposite, small, antonym-side inputs) stay 100%; downstream say-large nodes suppressed (panel values ~11%/10%/13% and 0%/0%/20%).

### Fig B4 — Operand swap in all three languages
- Slug: `multilingual-swap-operand-svg` → local `figures/multilingual_swap_operand_small_to_hot.svg`.
- Caption: "Interventions on the operand, swapping small to hot features in three different language input cases."
- Protocol: on the "small"/"petit"/"小" token, **small-size features −0.5×; hot-temperature features +1.5×** (hot features from an English prompt with "small" → "hot"). Strength axis 0×–1.5×.
- Outcomes: EN large → **cold**; FR grand → **f[roid]** (fr: cold); ZH 大 → **冷** (zh: cold). Node annotations: upstream ≈100% (ZH panel shows 110%/101%); say-large ≈0–12%; say-cold recruited.

### Fig B5 — Language swap
- Slug: `multilingual-swap-language-svg` → local `figures/multilingual_swap_output_language.svg`.
- Caption: "Interventions on language features in three different language input cases."
- Protocol: replace early **language-detection features** on the final token (equivariant `open-quote-in-language-X` + `beginning-of-document-in-language-Y` supernodes): **original language −5×, new language +6×**.
- Outcomes: EN→ZH: large → **大**; FR→EN: grand → **big**; ZH→FR: 大 → **grand**. Node annotations: antonym & small unchanged (100%); say-large-multilingual roughly preserved (~65–119%); original-language say-large suppressed (~18–39%); new-language say-large high (~76–105%). Demonstrates "we can edit the language while preserving the operation and operand of the computation."

### Fig B6 — The French circuit in more detail
- Slug: `french-multilignual-big-svg` → local `figures/multilingual_french_circuit_detail.svg`.
- Caption: "A slightly more detailed attribution graph for the French prompt, although still greatly simplified. Note that one of the most interesting interactions appears to be a QK-mediated effect, invisible to our present method (but validated in intervention experiments)."
- Content: tokens `Le | contr | aire | de " | petit | " est | "` → grand. Multi-token "contraire" **detokenized** into abstract multilingual features; `opposite (French)` → `antonym (multilingual)`; `petit` → `small (multilingual)`; a **"predict size" (multilingual)** feature group (elided from simplified diagrams; weaker effect); `quote (French)` language tracking (with linguistic cues also coming from other words); `say large (French)` + `large (multilingual)` → grand. The antonym→large edge is marked "**QK mediated interaction?**".

### Fig B7 — Cross-lingual feature overlap vs layer, by model scale
- Slug: `multilingual-intersection-svg` → local `figures/multilingual_feature_overlap_across_scale.svg`.
- Caption: "Analysis of shared feature usage between analogous prompts in different languages. Small model shares less features than Haiku between these more linguistically distant language pairs (English-Chinese and French-Chinese), compared to the English-French comparison (on right). On these more linguistically similar languages, both models share more, with a smaller gap."
- What it plots: three panels (**English-Chinese**, **French-Chinese**, **English-French** "Feature Intersection over Union"). X-axis: **Layer Depth** normalized 0.0–1.0; Y-axis: **Intersection Fraction** (ticks 0.1/0.2/0.3). Four curves per panel: Haiku, Smaller 18L model, Haiku baseline, Small baseline ("Baseline is unrelated prompts in same language pairs"; main curves are "Feature intersection between translated versions of same prompt").
- Shape: overlap low at beginning and end of the model, peaks in the middle; Haiku's mid-layer curve clearly above the 18L model's for EN-ZH and FR-ZH; gap smaller for EN-FR; baselines much lower.

## B.2 Key quantitative results
- **20 out of 27** features in the multilingual supernodes are active across all three prompts; only **10/27** appear in the pruned attribution graphs of all three prompts ("the set of features that are influential to the model's response varies quite a bit by prompt").
- Operation swap requires "unnatural strength": **+6×** the donor (synonym-prompt) activation; "the crossover point of when the intervention is effective is fairly consistent across languages (**about 4×**)".
- Intervention strengths used: operation −5×/+6×; operand −0.5×/+1.5×; language −5×/+6×.
- Feature-overlap experiment: IOU of feature sets (features active anywhere in context) per {paragraph, language pair, layer}, on Claude-generated FR/ZH translations of diverse paragraphs; mid-layer IOU for Haiku reaches roughly the 0.2–0.3 range (figure axis max 0.3), far above unrelated-paragraph baselines; Haiku > 18L model, with the largest gain on non-alphabet-sharing pairs (EN-ZH, FR-ZH).
- Synonym/antonym footnote: "all pairwise inner products between synonym and antonym encoder vectors are positive and the minimum decoder vector inner product is only slightly negative" (functionally opposite function vectors that are geometrically close).
- Supernode sizes visible in the HTML feature-ids: antonym = 6 features, synonym = 6, small-size = 3, hot-temperature = 3, French-language-detection = 3, Chinese-language-detection = 5 (h35 scan; e.g. antonym = b19418366, b20444397, b6808897, b8786631, b30636416, b5636530; `_6+_9` = b12406787; `_9+_9` = b24589983).

## B.3 Qualitative conclusions (precise statements)
1. **Shared-circuit claim:** the three prompts "are driven by very similar circuits, with shared multilingual components, and an analogous language-specific component" — language-invariant core + language-equivariant periphery. [Expected to generalize; the paper itself frames this as a general hypothesis about multilingual LMs.]
2. **Mechanism story:** "the model recognizes, using a language-independent representation, that it's being asked about antonyms of 'small'. This triggers antonym features, which mediate (via an effect on attention …) a map from small to large. In parallel with this, open-quote-in-language-X features track the language … and trigger the language-appropriate output feature" (e.g., "big"-in-Chinese). [Motif expected to generalize.]
3. **Three-part decomposition:** "We can think of this computation as involving three parts: operation (i.e. antonym), operand (i.e. small), and language", each independently intervenable. [Expected to generalize.]
4. **English as privileged default (model-specific!):** "there is a meaningful sense in which English is mechanistically privileged over other languages as the 'default'". Evidence: (a) "multilingual 'say large' features often have stronger direct effects to 'large' or 'big' in English as compared to other languages", with non-English outputs "more strongly mediated by say-X-in-language-Y features"; (b) "the English quote features have a weak and mixed direct effect on the English 'say large' features, instead having a double inhibitory effect" — they suppress features which themselves suppress "large" in English but promote "large" in other languages (example: an English-quote feature b20287395 whose strongest negative edge goes to a feature b1246381 that upweights "large" in Romance languages like French and downweights it in English). [For an English-trained Claude model; in a Qwen3 reproduction the *identity* of the default language may differ (plausibly English or Chinese given Qwen's training mix) — the transferable claim is that *some* language acts as a mechanistic default with others mediated by language-specific output features.]
5. **Layerwise profile:** "features at the beginning and end of models are highly language-specific (consistent with the {de,re}-tokenization hypothesis), while features in the middle are more language-agnostic." [Expected to generalize.]
6. **Scale claim:** "compared to the smaller model, Claude 3.5 Haiku exhibits a higher degree of generalization, and displays an especially notable generalization improvement for language pairs that do not share an alphabet (English-Chinese, French-Chinese)" — i.e., multilingual features "represent an increasing fraction of model representations with scale". [Requires ≥2 model scales to check; direction expected to generalize.]
7. **Generality across tasks:** "in the three simple prompts …, the key semantic transformation occurs using the same important nodes in every language, despite not sharing any tokens in the input."
8. **Method caveat (French circuit):** "One crucial interaction (between antonym and large) seems to be mediated by changing where attention heads attend, by participating in their QK circuits. This is invisible to our current approach, and might be seen as a kind of 'counterexample' concretely demonstrating a weakness of our present circuit analysis." [Important for a reproduction: the antonym→say-large edge may not appear as a direct edge in a frozen-attention attribution graph.]
9. **Literature positioning ("Do Models Think in English?"):** reconciles multilingual-feature evidence with Schut et al. (English-privileged) and Wendler et al. (multilingual-but-English-aligned): "Claude 3.5 Haiku is using genuinely multilingual features, especially in the middle layers. However, there are important mechanistic ways in which English is privileged … a multilingual representation in which English is the default output."

## B.4 Intervention experiments (summary table)
| # | Target | Source of donor features | Strengths | Outcome (EN / FR / ZH) |
|---|---|---|---|---|
| 1 | Operation: antonym → synonym (middle layers, final token) | English prompt `A synonym of "small" is "` | −5× antonym, +6× synonym; effective crossover ≈4× in all languages | little / min[uscule] / 微 — language-appropriate **synonyms**; say-large nodes suppressed, upstream unchanged |
| 2 | Operand: small → hot (early layers, operand token; features chosen by highest "graph influence") | English prompt with "hot" | −0.5× small-size, +1.5× hot-temperature | cold / f[roid] / 冷 — language-appropriate **antonyms of "hot"** |
| 3 | Language (early layers, final token: quote-in-language-X + beginning-of-document-in-language-Y) | target-language prompt | −5× original, +6× new | EN→ZH gives 大; FR→EN gives big; ZH→FR gives grand — **language changes, operation+operand preserved** |
- Cross-cutting takeaway stated in the paper: donor features come from **English** prompts in experiments 1–2, yet outputs stay language-appropriate ⇒ operation and operand components are language-independent.
- General caveat (paper, entity-recognition section but applies here): "the intervention strengths required to obtain interesting effects are quite high relative to the feature activations on the original prompts", suggesting identified features/connections "capture only a part of the story".

---

# C. Notes for checking a Qwen3-based reproduction

**Should transfer (qualitative motifs):**
- Addition: heterogeneous parallel pathways (approximate-magnitude + mod-10) rather than sequential carry; existence of lookup-table features (points in operand plots) and sum features (diagonals/modular stripes); "ends in N" modular features; introspection mismatch (model narrates the schoolbook carry algorithm); reuse of the same lookup features in non-arithmetic contexts (citations/tables); intermediate-result flagging in nested expressions.
- Multilingual: shared mid-layer semantic core with language-specific input (detokenization/quote/language-detection) and output (say-X-in-language-Y) periphery; independent editability of operation/operand/language; overlap peaking in middle layers; overlap increasing with model scale (needs two Qwen3 sizes, e.g. 0.6b vs 8b/14b).

**Model-specific (do NOT expect to match numerically):**
- All feature identities/indices, supernode compositions, percentage tables, exact intervention multiples (−5×/+6×, ≈4× crossover, −0.5×/+1.5×, −2×), 20/27 and 10/27 counts, 2,931-feature count (that one is for Anthropic's 18L research model, not Haiku), and 98.6%/54.8%/66.6% output probabilities.
- "English is the default language" — Claude-specific; for Qwen3 test *which* language is privileged rather than assuming English; Chinese is a live possibility.
- Tokenization-dependent details (e.g., "contr|aire" detokenization; 大/小 single-token behavior in the Qwen3 tokenizer; whether "calc: 36+59=" splits operands into single tokens).
- Claude-specific completion "large" vs "big" for the English prompt.

**Methodological details a faithful reproduction should mirror:**
- Operand plots: activity on the `=` token over all 10,000 (a,b) pairs, same geometric taxonomy.
- Interventions via constrained patching with an intervention layer; suppression as negative multiples of original activation; donor activations quoted as multiples of source-prompt activation; sweep strengths and report crossover.
- Cross-lingual IOU: same-text translations vs unrelated-text baseline, computed per layer over features active anywhere in the context.

# D. File inventory (local copies)

Digest: `(session-local, re-download via the URLs below) biology_digest.md`
Raw section text extracts: `notes/addition_section.txt`, `notes/multilingual_section.txt` (cleaned from HTML; [FIGURE]/[FEAT]/[FOOTNOTE] markers preserved). Raw HTML: `notes/biology.html`, `notes/methods.html`.

Figures in `notes/figures/` (SVG unless noted; source URL = `https://transformer-circuits.pub/2025/attribution-graphs/figma/<slug>.svg`):
- `addition_operand_plots.svg` (operand-plots-biology-svg)
- `addition_36_59_attribution_graph.svg` (36-59-svg)
- `addition_6plus9_feature_dataset_examples.png` (png/img_9ac0f261fe832699.png)
- `addition_polymer_journal_context.svg` (polymer-svg)
- `addition_polymer_journal_intervention.svg` (arithmetic-polymer-intervention-svg)
- `addition_intermediate_4plus5times3.svg` (addition-intermediate-svg)
- `multilingual_overview_part1.svg` (multilingual-overview-hover-1-svg)
- `multilingual_overview_part2.svg` (multilingual-overview-2-svg)
- `multilingual_swap_operation_antonym_to_synonym.svg` (multilingual-swap-operator-svg)
- `multilingual_swap_operand_small_to_hot.svg` (multilingual-swap-operand-svg)
- `multilingual_swap_output_language.svg` (multilingual-swap-language-svg)
- `multilingual_french_circuit_detail.svg` (french-multilignual-big-svg)
- `multilingual_feature_overlap_across_scale.svg` (multilingual-intersection-svg)
- `figures/png/` — 34 raster heatmaps/screenshots referenced by the SVGs (so they render locally). One background raster in the three swap SVGs is AccessDenied upstream; all data labels are vector text and unaffected.

Interactive attribution graphs (for node-by-node comparison): `https://transformer-circuits.pub/2025/attribution-graphs/static_js/attribution_graphs/index.html?slug=` + `calc-36-plus-59`, `order-of-operations-paren`, `polymer-add-9`, `opposite_of_small`, `opposite_of_petit`, `opposite_of_small_zh`.
