# llm-circuits

Mechanistic-interpretability research toolkit for LLM circuits.

## Philosophy

**circuit-tracer is used only to load transcoders.** We deliberately avoid
importing `ReplacementModel`, `AttributionGraph`, or any intervention machinery
from circuit-tracer. Instead, attribution graphs and interventions live under
`src/llm_circuits/circuits/` and will be implemented from scratch in this repo,
giving us full control over the computation graph and experiment loop.

## Quickstart

```bash
# Install everything (requires uv)
uv sync --all-groups

# Show environment info and available model specs
uv run llm-circuits info

# Inspect a transcoder config (lightweight, no big downloads)
uv run llm-circuits transcoder-inspect --size 0.6b

# Load transcoders (downloads weights)
uv run llm-circuits transcoder-load --size 0.6b

# Cache transcoders locally
uv run llm-circuits transcoder-cache --repo mwhanna/qwen3-0.6b-transcoders-lowl0 --cache-dir .cache/transcoders

# Generate text (pure Transformers, no circuit-tracer)
uv run llm-circuits generate --size 0.6b --prompt "The capital of France is"
```

## Project structure

```
src/llm_circuits/
  cli.py               # Typer CLI entry point
  settings.py           # Device / dtype / path defaults
  models/               # HF model loading (AutoModelForCausalLM)
  transcoders/          # Registry + circuit-tracer loader (the ONLY circuit-tracer import)
  instrumentation/      # Generic PyTorch hooks and activation recording
  circuits/             # Attribution graphs and interventions (our own impl)
configs/                # YAML configs per model size
examples/               # Runnable scripts
tests/                  # pytest suite
notebooks/              # Research notebooks
```

## Available model specs

| Size | HF Model ID | Transcoder Repo |
|------|-------------|-----------------|
| 0.6b | Qwen/Qwen3-0.6B | mwhanna/qwen3-0.6b-transcoders-lowl0 |
| 1.7b | Qwen/Qwen3-1.7B | mwhanna/qwen3-1.7b-transcoders-lowl0 |
| 4b   | Qwen/Qwen3-4B   | mwhanna/qwen3-4b-transcoders |
| 8b   | Qwen/Qwen3-8B   | mwhanna/qwen3-8b-transcoders |
| 14b  | Qwen/Qwen3-14B  | mwhanna/qwen3-14b-transcoders-lowl0 |

## Development

```bash
# Lint
make lint

# Format
make format

# Test
make test

# All checks
make check
```
