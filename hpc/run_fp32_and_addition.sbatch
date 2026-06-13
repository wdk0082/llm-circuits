#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=llm-fp32-addition
#SBATCH --output=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.out
#SBATCH --error=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

REPO="$HOME/llm-circuits"
PATH="$HOME/.local/bin:$PATH"

cd "$REPO"

# Load env vars (caches, tokens, etc.) from .env
set -a; source .env; set +a

# Reproduce environment from lockfile (won't mutate it)
uv sync --frozen

echo "########################################################################"
echo "# STEP 3: fp32 vs original drift (local replacement faithfulness)"
echo "########################################################################"
uv run --frozen python examples/compare_local_replacement_model.py

echo "########################################################################"
echo "# STEP 7: addition circuit (build -> prune -> identify -> ablate)"
echo "########################################################################"
uv run --frozen python examples/addition_circuit.py

echo "ALL DONE"
