#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=llm-addition-circuit
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

echo "# Addition circuit: build -> prune -> identify -> ablate"
uv run --frozen python examples/addition_circuit.py
echo "ALL DONE"
