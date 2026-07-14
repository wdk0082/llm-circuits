#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=llm-supernode-inputs
#SBATCH --output=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.out
#SBATCH --error=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.err

# Stage 1 (GPU) of the supernode pipeline for the ADDITION v3 reproduction: derive the
# studied/donor pairs from a fresh accuracy grid, dump the four pruned addition graphs
# (first/ones/donor/reuse), the three-probe operand grids (grids.npz), and the
# reuse-moment activations under $LLM_CIRCUITS_ARTIFACTS_DIR/supernode_inputs/<size>/.
# Multilingual dumps are skipped (that reproduction is finalized; restore the
# repro-dumps-4b tarball if its review pages are ever needed on this cluster).

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

REPO="$HOME/llm-circuits"
PATH="$HOME/.local/bin:$PATH"

cd "$REPO"
set -a; source .env; set +a
uv sync --frozen

echo "# build_supernode_inputs.py (addition only) on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null))"
uv run --frozen python notebooks/build_supernode_inputs.py --size 4b --skip-multilingual
echo "ALL DONE"
