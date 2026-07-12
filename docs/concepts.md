# Concepts

This page maps the mechanistic-interpretability pipeline onto the modules that
implement it. It is the conceptual companion to the {doc}`demo walkthrough
<guide/demo>`, which runs the same steps end to end on a single prompt.

## Transcoders

A **transcoder** is a wide, sparse approximation of an MLP layer: it encodes the
MLP input into many interpretable **features**, then decodes them back to the
MLP output. Replacing an MLP with its transcoder swaps a dense, hard-to-read
computation for a sparse one whose active features can be labelled.

The package supports the Qwen3 family. Model keys, HF ids, and transcoder repos
live in a frozen registry ({class}`~llm_circuits.transcoders.registry.ModelSpec`),
and transcoders are fetched by
{func}`~llm_circuits.transcoders.circuit_tracer_loader.load_transcoder` — the
**only** function in the package that imports circuit-tracer. Per-feature labels
(top/bottom output logits, activating examples) come from
{mod}`~llm_circuits.transcoders.feature_labels`.

```{admonition} Lazy vs. eager decoders
:class: note
`load_transcoder(..., lazy_decoder=True)` re-reads `W_dec` from disk on every
decode — fine for encode-only work, ~500× slower for repeated forwards. Steering
and attribution jobs should load **eagerly** (`lazy_decoder=False`). See the
performance notes in `CLAUDE.md`.
```

## The replacement model

Swapping every MLP for its transcoder reconstruction gives a **replacement
model**. There are two flavours, and the difference is the whole point:

**Raw swap** ({func}`~llm_circuits.circuits.replacement_model.replace_mlps_with_transcoders`)
replaces MLP outputs with transcoder reconstructions during the forward pass.
Because transcoders are approximate, this is **lossy** —
{func}`~llm_circuits.circuits.replacement_model.compare_models` quantifies the
gap (KL divergence, logit cosine, top-1 agreement).

**Local replacement** ({func}`~llm_circuits.circuits.local_replacement_model.run_local_replacement`)
makes the swap **exact** for one specific input by, additionally:

- **adding error nodes** — the reconstruction error
  (`original_mlp_out − reconstruction`) is injected back into the residual
  stream, so the final logits match the original model bit-for-bit (up to
  round-off);
- **freezing attention weights and RMSNorm denominators** — this linearises the
  model in the residual stream, leaving the transcoders as the only
  nonlinearity;
- **detaching reconstructions** — features and error nodes become constants, so
  gradients flow only through the linear residual pathway.

This exact, linearised model is what the attribution graph is built on. The
first token position (the attention sink / BOS) is preserved unswapped —
transcoders cannot reliably reconstruct it — which is why the API threads an
`n_bos_tokens` argument throughout.

## Attribution graphs

On the linearised local replacement model, the influence of one node on another
is a gradient. {func}`~llm_circuits.circuits.attribution_graph.build_attribution_graph`
back-propagates through the linearised residual stream with
{func}`torch.autograd.grad` to produce an
{class}`~llm_circuits.circuits.attribution_graph.AttributionGraph`:

- **nodes** — the input embedding, transcoder **features**, **error** nodes, and
  the output **logits**;
- **edges** — the (signed) direct influence of each upstream node on each
  downstream node.

The raw graph is dense. {func}`~llm_circuits.circuits.graph_pruning.prune_graph`
keeps only the influential subgraph — a `node_threshold` on cumulative node
influence and an `edge_threshold` on cumulative edge weight — producing the
compact graph a human actually reads.
{func}`~llm_circuits.circuits.graph_pruning.graph_to_dict` /
{func}`~llm_circuits.circuits.graph_pruning.graph_from_dict` round-trip it to
JSON.

## Interventions

To test a hypothesis, steer or ablate the features that a graph implicates.
{func}`~llm_circuits.circuits.interventions.steer` builds an intervention on a
`(layer, feature)` using the package's **additive-delta (`m`) convention**: a
feature's new activation is `(1 + m) · clean`, so the applied delta is
`m · clean · W_dec`. Thus `m = 0` is a no-op, `m = −1` **ablates**, and
`m = −2` **flips** the feature. Deltas are computed once from the clean
activations and applied while the real model runs (attention patterns frozen).

There are two modes, and the demo runs the *same* steer through both so they are
directly comparable:

**Propagate mode** (`patch_end_layer=None`) — the perturbation flows through the
real model from the steered layer onward. This matches circuit-tracer's
unconstrained clamp for single-layer or causally-uncoupled steers.

**Constrained mode** (`patch_end_layer=L`) — activations at layers `≤ L` are
pinned to their perturbed values and the real model runs above `L`. `L` is the
paper's layer-range knob; {func}`~llm_circuits.circuits.interventions.sweep_patch_end_layer`
sweeps it. This mode is verified numerically equivalent to circuit-tracer's
`feature_intervention` (see the DEVLOG and `verification/`).

Effects are read out with
{func}`~llm_circuits.circuits.interventions.ablation_prob_effect` and
{func}`~llm_circuits.circuits.interventions.ablation_logit_effect` (before/after
probability and logit of a target token).

```{admonition} Protocol scope
:class: warning
Fixed-delta steering (`value − clean`) is a deliberate protocol choice. For
multi-layer *coupled* stacks it is **not** bit-identical to circuit-tracer's
propagate-mode clamp, which re-anchors each delta on the perturbed pass
(`value − current`). The constrained mode *is* verified equivalent. The full
argument is in `DEVLOG_EXTRA.md` §3.1 and the docstring of
{mod}`~llm_circuits.circuits.interventions`.
```

## Visualisation

{func}`~llm_circuits.circuits.graph_explorer.render_graph_explorer_html` renders
a pruned graph dict into a single self-contained HTML page that reproduces
Anthropic's circuit viewer: the position×layer graph, a node detail panel
(input/output features, token predictions, activating examples), and manual
supernode grouping with subgraph collapse. No server or model is needed to view
it. A lighter static renderer lives in
{mod}`~llm_circuits.circuits.visualization`.

For a **live** version of build → prune → steer → sweep in the browser, run the
interactive UI: `uv run --group serve llm-circuits serve` (see the
{doc}`CLI reference <cli>`).
