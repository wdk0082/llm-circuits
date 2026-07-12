# The demo walkthrough

`examples/demo.py` runs the whole toolkit end to end on a single `a+b=` prompt —
the same calls `notebooks/multilingual.ipynb` makes, minus the supernode/swap
protocol. It is the fastest way to see every API in context. This page walks
through its five steps; the {doc}`concepts <../concepts>` page explains the ideas
behind them.

## Running it

```bash
uv run python examples/demo.py                 # full walkthrough, Qwen3-4b
uv run python examples/demo.py --size 1.7b --problem 2+3
uv run python examples/demo.py --lazy-decoder  # low VRAM, much slower
```

The 0.6B model cannot add reliably, so the demo defaults to **Qwen3-4B** and
first searches a few prompt formats for one the model actually solves. Run it on
a GPU node (`sbatch hpc/run_demo.sbatch`). Artifacts (JSON graph, explorer HTML,
sweep PNG) land in `artifacts/demo/`.

## The pipeline, step by step

The five steps map one-to-one onto the package API.

### 1 · Load

Load the model and its matching transcoders. The size key resolves through the
registry, so the model and transcoders are guaranteed to correspond. **Eager**
decoder loading matters here — the sweep in step 5 runs one intervention per end
layer, and a lazy decoder re-reads `W_dec` from disk on every decode.

```python
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

model, tokenizer = load_qwen3("4b", dtype_str="bf16", device_map=device)
model.eval()
loaded = load_transcoder("qwen3-4b", device=device, dtype=dtype, lazy_decoder=False)
tc = loaded.transcoder
```

The prompt is turned into token ids with the Qwen3 chat template via
{func}`~llm_circuits.instrumentation.chat.prepare_messages`, which also reports
`n_bos` — the number of leading positions (the attention sink) that the
replacement model must leave unswapped.

### 2 · Replace

Two replacement models, to show what error nodes buy. The **raw swap** is lossy;
{func}`~llm_circuits.circuits.replacement_model.compare_models` measures the gap.
Adding error nodes and freezing attention/LayerNorm
({func}`~llm_circuits.circuits.local_replacement_model.run_local_replacement`)
reproduces the real logits exactly — and that exact, linearised model is what
the graph is built on.

```python
from llm_circuits.circuits.replacement_model import compare_models
from llm_circuits.circuits.local_replacement_model import run_local_replacement

cmp = compare_models(model, tc, input_ids, n_bos_tokens=n_bos)   # lossy
# cmp.kl_divergence, cmp.cosine_similarity, cmp.top1_agreement

lctx = run_local_replacement(model, tc, input_ids, n_bos_tokens=n_bos)  # exact
# max |lctx.logits - lctx.original_logits| ~ bf16 round-off
```

### 3 · Graph → prune → explore

Build the attribution graph, prune it to a readable subgraph, label the surviving
features, and render the interactive explorer.

```python
from llm_circuits.circuits.attribution_graph import build_attribution_graph
from llm_circuits.circuits.graph_pruning import prune_graph, graph_to_dict
from llm_circuits.circuits.graph_explorer import render_graph_explorer_html

graph = build_attribution_graph(model, tc, input_ids, n_bos_tokens=n_bos)
pruned = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)

gd = graph_to_dict(pruned.graph, influence_scores=pruned.influence_scores, ...)
render_graph_explorer_html(gd, "artifacts/demo/explorer.html")
```

The demo then selects the features to steer **semantically**, not by influence
rank: the answer-writing features (those whose top output logits include the
answer token) that sit below a depth cap. That discipline — the same one the
notebook's supernodes use — is what leaves a real layer range for the step-5
sweep to move. See the extended comment in the source below.

### 4 · Propagate

Steer the selected features and let the edit flow through the real model
(`patch_end_layer=None`). {func}`~llm_circuits.circuits.interventions.steer`
builds each intervention with the `m` convention (`m = −2` flips the feature);
{func}`~llm_circuits.circuits.interventions.ablation_prob_effect` reads the
before/after probability of the answer.

```python
from llm_circuits.circuits.interventions import steer, run_feature_intervention, ablation_prob_effect

steers = [steer(nd.layer, nd.feature_idx, m=-2.0, position=nd.position) for nd in top]
prop = run_feature_intervention(model, tc, input_ids, steers, n_bos_tokens=n_bos)
p0, p_prop = ablation_prob_effect(prop, [answer_id])[answer_id]
```

### 5 · Constrained + sweep

Run the *same* steer in **constrained mode**: pin activations at layers `≤ L`,
run the real model above `L`, and sweep `L` — the paper's layer-range knob.
{func}`~llm_circuits.circuits.interventions.sweep_patch_end_layer` returns the
per-layer probabilities and the most suppressive end layer, which the demo plots
against the propagate baseline.

```python
from llm_circuits.circuits.interventions import sweep_patch_end_layer

sw = sweep_patch_end_layer(model, tc, input_ids, steers, answer_id, n_bos_tokens=n_bos)
# sw.end_layers, sw.probs, sw.best_end_layer  ->  artifacts/demo/sweep.png
```

Because steps 4 and 5 steer the same features, the final printout isolates what
constrained patching buys over letting the edit propagate.

## Full source

```{literalinclude} ../../examples/demo.py
:language: python
:caption: examples/demo.py
:linenos:
```

## Where to go next

- {doc}`../concepts` — the ideas behind each step.
- {doc}`../api/index` — the full API reference for every function used here.
- {doc}`multilingual` — the research reproduction that builds on this machinery
  with the supernode/swap protocol.
- The **live** version of steps 3–5 in the browser:
  `uv run --group serve llm-circuits serve` ({doc}`CLI reference <../cli>`).
```
