#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=llm-multilingual-notebook
#SBATCH --output=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.out
#SBATCH --error=/rds/user/%u/hpc-work/llm-circuits/logs/%x-%j.err

# Execute notebooks/multilingual.ipynb on a GPU node (reproduces the multilingual dive A-D).
# --allow-errors so a single failing cell doesn't abort the rest; figures + graph explorers
# save under artifacts/multilingual_notebook/.
set -euo pipefail
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
REPO="$HOME/llm-circuits"; PATH="$HOME/.local/bin:$PATH"
cd "$REPO"; set -a; source .env; set +a
uv sync --frozen --group notebook

echo "# Executing notebooks/multilingual.ipynb on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null))"
cd "$REPO/notebooks"
uv run --frozen --group notebook jupyter nbconvert --to notebook --inplace --allow-errors \
  --ExecutePreprocessor.timeout=3600 --execute multilingual.ipynb
echo "ALL DONE"
