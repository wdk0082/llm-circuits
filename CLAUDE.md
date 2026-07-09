# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mechanistic-interpretability research toolkit for analyzing LLM circuits, supporting the Qwen3 model family with transcoder support. Uses Python 3.11, managed with `uv`.

**Key design principle:** circuit-tracer is used **only** to load transcoders. We deliberately avoid importing `ReplacementModel`, `AttributionGraph`, or any intervention machinery from circuit-tracer. Attribution graphs and interventions are implemented from scratch under `src/llm_circuits/circuits/` for full control over the computation graph and experiment loop.

## HPC Environment (**Very important**)

- **Queue System:** Slurm
- **Current Session:** This session is running on a dedicated GPU compute node (Ampere).
- **Project Account:** MPHIL-DIS-SL2-GPU
- **Usage Policy:** All heavy computations, model training, or indexing must be done within this interactive session. Do not run heavy processes on the login nodes.

**Always 1. load `.env` before any bash commands (by `set -a; source .env; set +a`) and 2. use `uv` when running python commands.**

## Cloud TPU workflow (`gcp/`)

An alternative to HPC/Slurm: run on an ephemeral Google Cloud TPU (v6e). The
TPU is **disposable compute; durable state lives in a GCS bucket.** All scripts
run on your laptop and read config from `.env` (via `gcp/lib.sh`).

- **Lifecycle:** `gcp/setup_storage.sh` (one-time IAM grant) → `gcp/create.sh`
  (Spot + queued resource, then `bootstrap.sh`) → `gcp/launch.sh <script.py>`
  → `gcp/pull.sh` → `gcp/teardown.sh`. Helpers: `gcp/status.sh`, `gcp/ssh.sh`.
- **`bin/run`** is the run wrapper (sources `.env`, prepends `.venv/bin` to
  `PATH`, execs) used on both laptop and VM; `launch.sh` invokes
  `./bin/run python -u <script.py>` on the TPU.
- **Device:** `launch.sh` injects `LLM_CIRCUITS_DEVICE=tpu`, which
  `settings.default_device()` resolves to the torch_xla `"xla"` device;
  `hf_loader` loads on CPU then moves to XLA. Smoke test:
  `gcp/launch.sh examples/tpu_smoke_test.py`.
- **torch_xla** is installed on the VM by `gcp/bootstrap.sh`
  (`torch_xla[tpu]==2.9.0` + matching `torch==2.9.0` — torch_xla lags torch, so
  the VM's torch is downgraded from the lockfile's 2.10), **not** in
  `pyproject.toml`, so the lockfile stays cross-platform.
- Full details, projects, and cross-project auth: `gcp/README.md`.

## Architecture

- **`src/llm_circuits/`** — src-layout package
  - **`cli.py`** — Typer CLI entry point (`llm-circuits` command)
  - **`settings.py`** — Device/dtype/path defaults (auto-detects CUDA/MPS/CPU, bf16 for CUDA/CPU, fp32 for MPS)
  - **`models/`** — HF model loading via `AutoModelForCausalLM`; `qwen3.py` has family-specific wrappers
  - **`transcoders/`** — `registry.py` has frozen `ModelSpec` dataclass mapping family-prefixed keys (e.g. `qwen3-0.6b`, `qwen3-4b`) to HF repos; `circuit_tracer_loader.py` is the **only** file that imports from circuit-tracer
  - **`instrumentation/`** — Generic PyTorch hook utilities (`attach_hook` context manager, `ActivationRecorder`)
  - **`circuits/`** — Custom attribution graphs (`build_attribution_graph`), local/global replacement models, graph pruning, HTML visualization, and feature interventions (`run_feature_intervention` = circuit-tracer's `feature_intervention` on the real model: decoder delta with the M convention, M=0 no-change / -1 ablate / -2 flip; cross-verified in `verification/`).
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
