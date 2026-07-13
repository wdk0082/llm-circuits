# llm-circuits

[![Documentation Status](https://readthedocs.org/projects/llm-circuits/badge/?version=latest)](https://llm-circuits.readthedocs.io/en/latest/?badge=latest)

Mechanistic-interpretability research toolkit for analysing **LLM circuits** in the
Qwen3 model family. It reproduces Anthropic's circuit-tracing method — from
[*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
and the [`circuit-tracer`](https://github.com/safety-research/circuit-tracer) library —
reimplementing attribution graphs, replacement models, and feature interventions from
scratch, and using `circuit-tracer` **only** to load transcoders (never its
`ReplacementModel`, `AttributionGraph`, or intervention machinery).

## Project structure

```
src/llm_circuits/
  cli.py                # Typer CLI entry point
  settings.py           # Device / dtype / path defaults
  models/               # HF model loading (AutoModelForCausalLM)
    qwen3.py            # Qwen3-specific convenience loader
  transcoders/          # Registry + circuit-tracer loader (the ONLY circuit-tracer import)
  instrumentation/      # Generic PyTorch hooks and activation recording
  circuits/             # Attribution graphs and interventions (our own impl)
  serve/                # Interactive UI (FastAPI + static frontend)
examples/               # demo.py (one end-to-end toolkit walkthrough) + tpu_smoke_test.py
notebooks/              # The paper reproductions: <behavior>.ipynb + <behavior>_helper.py
docs/                   # Sphinx documentation (guides + autodoc API reference)
tests/                  # pytest suite
```

## Documentation

Full API docs and guides are built with [Sphinx](https://www.sphinx-doc.org/)
(sources in [`docs/`](docs/)):

```bash
make docs      # -> docs/_build/html/index.html
```

The docs are organised around the two worked examples — a walkthrough of
`examples/demo.py` and the rendered `notebooks/multilingual.ipynb` — plus a
concepts guide, a CLI reference, and the autodoc API reference. See
[`docs/index.md`](docs/index.md) for the entry point.

## Interactive UI

A FastAPI app (`src/llm_circuits/serve/`) for **live** circuit work from the browser:
load a Qwen3 size, build + re-prune attribution graphs, steer features, and sweep the
constrained-patching end layer — the interactive version of `examples/demo.py`.

```bash
uv run --group serve llm-circuits serve          # GPU node; open http://localhost:8000
uv run --group serve llm-circuits serve --mock   # CPU mock engine (UI dev, no model)
```

(`--group serve` is needed at *run* time — plain `uv run` re-syncs the env without the
serve extras.)

On the HPC, run it inside an interactive GPU allocation with
`bash hpc/run_interactive_server.sh` — VS Code auto-forwards the port. The server
binds `127.0.0.1` on purpose; forward the port rather than exposing it.

## Available model specs

| Key | Family | Size | HF Model ID | Transcoder Repo | Transcoder Type |
|-----|--------|------|-------------|-----------------|-----------------|
| qwen3-0.6b | qwen3 | 0.6b | Qwen/Qwen3-0.6B | mwhanna/qwen3-0.6b-transcoders-lowl0 | per-layer |
| qwen3-1.7b | qwen3 | 1.7b | Qwen/Qwen3-1.7B | mwhanna/qwen3-1.7b-transcoders-lowl0 | per-layer |
| qwen3-4b | qwen3 | 4b | Qwen/Qwen3-4B | mwhanna/qwen3-4b-transcoders | per-layer |
| qwen3-8b | qwen3 | 8b | Qwen/Qwen3-8B | mwhanna/qwen3-8b-transcoders | per-layer |
| qwen3-14b | qwen3 | 14b | Qwen/Qwen3-14B | mwhanna/qwen3-14b-transcoders-lowl0 | per-layer |

## Links

- **Documentation** — <https://llm-circuits.readthedocs.io>
- **circuit-tracer** (used only to load transcoders) — <https://github.com/safety-research/circuit-tracer>
- **Paper** — *On the Biology of a Large Language Model* — <https://transformer-circuits.pub/2025/attribution-graphs/biology.html>
- **Paper** — *Circuit Tracing: Revealing Computational Graphs in Language Models* — <https://transformer-circuits.pub/2025/attribution-graphs/methods.html>
- **Transcoders** — per-layer Qwen3 transcoders (mwhanna, Hugging Face) — <https://huggingface.co/mwhanna>
- **Models** — Qwen3 (Hugging Face) — <https://huggingface.co/Qwen>
- **Transformer Circuits Thread** — <https://transformer-circuits.pub/>
