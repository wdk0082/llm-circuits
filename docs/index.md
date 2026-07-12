# llm-circuits

Mechanistic-interpretability research toolkit for analysing **LLM circuits** in
the Qwen3 model family, built around per-layer **transcoders**.

```{admonition} Design principle
:class: tip
**circuit-tracer is used _only_ to load transcoders.** Attribution graphs,
replacement models, and interventions are implemented from scratch under
`llm_circuits.circuits` for full control over the computation graph and the
experiment loop. The toolkit never imports circuit-tracer's `ReplacementModel`,
`AttributionGraph`, or intervention machinery.
```

## What's here

The package reproduces the core of Anthropic's circuit-tracing method as a
small, inspectable library. The pipeline — and the API — reads top to bottom:

- **Load** a Qwen3 model and its matching transcoders from a
  {doc}`registry <api/transcoders>`.
- **Replace** MLPs with transcoder reconstructions — the
  {doc}`replacement model <api/circuits>`. Add error nodes and freeze
  attention/LayerNorm and the real logits are reproduced exactly.
- **Attribute**: build an attribution graph by autograd on the linearised
  residual stream, then **prune** it to a human-readable subgraph.
- **Intervene**: steer or ablate features — *propagate mode* vs. *constrained*
  layer-range patching — and measure the effect on the answer.
- **Visualise**: render a self-contained interactive graph explorer (HTML).

See {doc}`concepts` for how these pieces fit together, module by module.

## Two worked examples

The documentation is built around two files you can read and run:

**{doc}`The demo walkthrough <guide/demo>`** — `examples/demo.py`, one
end-to-end pass over the whole pipeline on a single `a+b=` prompt. The fastest
way to see every API call in context: load → replace → graph → prune → intervene
→ sweep.

**{doc}`The multilingual reproduction <guide/multilingual>`** —
`notebooks/multilingual.ipynb`, the paper-exact reproduction of multilingual
antonym circuits on Qwen3-4B: attribution graphs, supernodes, three feature
swaps, cross-lingual overlap, and a 4b-vs-8b scale comparison.

## Install

```bash
uv sync --all-groups          # or: make install
uv run llm-circuits info      # environment + available model specs
```

Full details in {doc}`installation`.

```{toctree}
:hidden:
:caption: Getting started

installation
concepts
```

```{toctree}
:hidden:
:caption: Examples

guide/demo
guide/multilingual
```

```{toctree}
:hidden:
:caption: Reference

cli
api/index
```
