# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mechanistic-interpretability research toolkit for analyzing LLM circuits, supporting Qwen3 and Gemma2 models with transcoder support. Uses Python 3.11, managed with `uv`.

**Key design principle:** circuit-tracer is used **only** to load transcoders. We deliberately avoid importing `ReplacementModel`, `AttributionGraph`, or any intervention machinery from circuit-tracer. Attribution graphs and interventions are implemented from scratch under `src/llm_circuits/circuits/` for full control over the computation graph and experiment loop.

## Commands

```bash
make install   # uv sync --all-groups
make lint      # uv run ruff check src tests examples
make format    # uv run ruff format + ruff check --fix
make test      # uv run pytest
make check     # lint + test
make all       # format + check
```

Run a single test file:
```bash
uv run pytest tests/test_registry.py -v
```

Run the CLI:
```bash
uv run llm-circuits info
uv run llm-circuits generate --size 0.6b --prompt "Hello"
```

## Architecture

- **`src/llm_circuits/`** — src-layout package
  - **`cli.py`** — Typer CLI entry point (`llm-circuits` command)
  - **`settings.py`** — Device/dtype/path defaults (auto-detects CUDA/MPS/CPU, bf16 for CUDA/CPU, fp32 for MPS)
  - **`models/`** — HF model loading via `AutoModelForCausalLM`; `qwen3.py` and `gemma2.py` have family-specific wrappers
  - **`transcoders/`** — `registry.py` has frozen `ModelSpec` dataclass mapping family-prefixed keys (e.g. `qwen3-0.6b`, `gemma2-2b`) to HF repos; `circuit_tracer_loader.py` is the **only** file that imports from circuit-tracer
  - **`instrumentation/`** — Generic PyTorch hook utilities (`attach_hook` context manager, `ActivationRecorder`)
  - **`circuits/`** — Custom attribution graphs and interventions (WIP); `ablate_module()` context manager for zero-ablation
  - **`utils/`** — Path resolution, config loading, `seed_everything`
- **`configs/`** — YAML configs per model size
- **`examples/`** — Runnable scripts (`uv run python examples/<script>.py`)

## Code Style

- Ruff for linting and formatting: line-length 100, target Python 3.11
- Rule sets: E, F, I, W, UP, B, SIM, RUF (E501 ignored)
- Uses `from __future__ import annotations` throughout
- Pre-commit hooks configured for ruff

## Environment

- `HF_TOKEN` required for gated models (see `.env.example`)
- `LLM_CIRCUITS_DEVICE` overrides auto-detected device

## Additional Requirements

- Always run the `.github/workflows/ci.yml` to check CIs.
- 