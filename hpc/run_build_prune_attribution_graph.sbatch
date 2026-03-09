#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=llm-run-1gpu
#SBATCH --output=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.out
#SBATCH --error=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh                # copy from the example script
module purge
module load rhel8/default-amp

# --- Paths: code in HOME, heavy stuff in RDS ---
REPO="$HOME/llm-circuits"                      # <-- change this
WORK="$HOME/rds/hpc-work/llm-circuits"               # symlinked to /rds/user/.../hpc-work/llm
PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WORK"/{logs,artifacts,checkpoints,.cache/uv,.cache/huggingface,.cache/wandb,.venv}

cd "$REPO"

# Load env vars (caches, tokens, etc.) from .env
set -a; source .env; set +a

# Reproduce environment from lockfile (won't mutate it)
uv sync --frozen

# Run training (single process, single GPU)
uv run --frozen python examples/build_attribution_graph.py
uv run --frozen python examples/prune_attribution_graph.py
