#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --job-name=llm-addition-notebook
#SBATCH --output=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.out
#SBATCH --error=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.err

# Execute notebooks/addition.ipynb on a GPU node (the paper-exact addition reproduction;
# helpers in notebooks/addition_helper.py). --allow-errors so a single failing cell
# doesn't abort the rest — errors land inline in the executed notebook for diagnosis;
# artifacts (explorer HTMLs, PNGs, JSONs) save under artifacts/paper_addition/<size>/.

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

REPO="$HOME/llm-circuits"
PATH="$HOME/.local/bin:$PATH"

cd "$REPO"
set -a; source .env; set +a
uv sync --frozen --group notebook   # jupyterlab (-> nbconvert/nbformat) + ipykernel

echo "# Executing notebooks/addition.ipynb on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null))"
cd "$REPO/notebooks"
uv run --frozen --group notebook jupyter nbconvert --to notebook --inplace --allow-errors \
  --ExecutePreprocessor.timeout=3000 --execute addition.ipynb
echo "ALL DONE"
