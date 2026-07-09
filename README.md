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
make install

# Show environment info and available model specs
uv run llm-circuits info

# Inspect a transcoder config (lightweight, no big downloads)
uv run llm-circuits transcoder-inspect --size qwen3-0.6b

# Load transcoders (downloads weights)
uv run llm-circuits transcoder-load --size qwen3-0.6b

# Cache transcoders locally
uv run llm-circuits transcoder-cache --repo mwhanna/qwen3-0.6b-transcoders-lowl0 --cache-dir .cache/transcoders

# Generate text (pure Transformers, no circuit-tracer)
uv run llm-circuits generate --key qwen3-0.6b --prompt "The capital of France is"
```

> **Backward compatibility:** Bare size keys like `--size 0.6b` still work and
> resolve to the corresponding Qwen3 entry.

## Project structure

```
src/llm_circuits/
  cli.py               # Typer CLI entry point
  settings.py           # Device / dtype / path defaults
  models/               # HF model loading (AutoModelForCausalLM)
    qwen3.py            # Qwen3-specific convenience loader
  transcoders/          # Registry + circuit-tracer loader (the ONLY circuit-tracer import)
  instrumentation/      # Generic PyTorch hooks and activation recording
  circuits/             # Attribution graphs and interventions (our own impl)
configs/                # YAML configs per model size
examples/               # Runnable machinery demos (not the paper reproduction)
tests/                  # pytest suite
notebooks/              # The paper reproductions: <behavior>.ipynb + <behavior>_helper.py
```

## Available model specs

| Key | Family | Size | HF Model ID | Transcoder Repo | Transcoder Type |
|-----|--------|------|-------------|-----------------|-----------------|
| qwen3-0.6b | qwen3 | 0.6b | Qwen/Qwen3-0.6B | mwhanna/qwen3-0.6b-transcoders-lowl0 | per-layer |
| qwen3-1.7b | qwen3 | 1.7b | Qwen/Qwen3-1.7B | mwhanna/qwen3-1.7b-transcoders-lowl0 | per-layer |
| qwen3-4b | qwen3 | 4b | Qwen/Qwen3-4B | mwhanna/qwen3-4b-transcoders | per-layer |
| qwen3-8b | qwen3 | 8b | Qwen/Qwen3-8B | mwhanna/qwen3-8b-transcoders | per-layer |
| qwen3-14b | qwen3 | 14b | Qwen/Qwen3-14B | mwhanna/qwen3-14b-transcoders-lowl0 | per-layer |

## Cloud TPU

Run experiments on an ephemeral Google Cloud TPU (v6e) via the lifecycle
scripts in [`gcp/`](gcp/README.md). The TPU is disposable compute; durable
state lives in a GCS bucket. All scripts run on your laptop and read config
from `.env`.

```bash
cp .env.example .env             # fill in CRSID / GIT_REMOTE / GCS_BUCKET
gcp/setup_storage.sh             # one-time: grant the TPU access to your bucket
gcp/create.sh                    # provision (Spot + queued resource) and bootstrap
gcp/launch.sh examples/addition_circuit.py   # run on the TPU (LLM_CIRCUITS_DEVICE=tpu)
gcp/pull.sh                      # bring artifacts back to ./artifacts
gcp/teardown.sh                  # delete the TPU; bucket data is kept
```

`torch_xla` is installed on the VM by `gcp/bootstrap.sh` (kept out of
`pyproject.toml` so the lockfile stays cross-platform). See
[`gcp/README.md`](gcp/README.md) for the full lifecycle and cross-project auth.

## Development

All development tasks are available as Makefile targets:

```bash
make install   # Install all dependencies (uv sync --all-groups)
make lint      # Lint with ruff (src, tests, examples)
make format    # Auto-format and fix with ruff (src, tests, examples)
make test      # Run pytest
make check     # Run lint + test
make all       # Run format + check (format, lint, then test)
make clean     # Remove __pycache__, .mypy_cache, .pytest_cache, .ruff_cache, build artifacts
```
