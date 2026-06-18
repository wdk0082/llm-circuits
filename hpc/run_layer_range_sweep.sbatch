#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=llm-layer-range-sweep
#SBATCH --output=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.out
#SBATCH --error=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

REPO="$HOME/llm-circuits"
PATH="$HOME/.local/bin:$PATH"

cd "$REPO"
set -a; source .env; set +a
uv sync --frozen

echo "# Constrained-patching layer-range sweep (patch end layer; reuses built addition graphs)"
uv run --frozen python -u examples/layer_range_sweep.py
echo "ALL DONE"
