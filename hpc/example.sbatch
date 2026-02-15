#!/bin/bash
#SBATCH --account=<PROJECT>-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --job-name=llm-train-1gpu
#SBATCH --output=/rds/user/%u/hpc-work/llm/logs/%x-%j.out
#SBATCH --error=/rds/user/%u/hpc-work/llm/logs/%x-%j.err

set -euo pipefail

# --- Paths: code in HOME, heavy stuff in RDS ---
REPO="$HOME/your-repo"                      # <-- change this
WORK="$HOME/rds/hpc-work/llm"               # symlinked to /rds/user/.../hpc-work/llm
mkdir -p "$WORK"/{logs,checkpoints,.cache/uv,.cache/huggingface,.cache/wandb,.venvs}

cd "$REPO"

# Load env vars (caches, tokens, etc.) from .env
set -a; source .env; set +a

# Reproduce environment from lockfile (won't mutate it)
uv sync --frozen

# Run training (single process, single GPU)
uv run --frozen python train.py \
  --data_dir "$WORK/data" \
  --output_dir "$WORK/checkpoints/run-${SLURM_JOB_ID}"
