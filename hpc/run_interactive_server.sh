#!/bin/bash
# Launch the interactive attribution-graph + steering server ON a GPU node.
#
# Intended for an *interactive* Slurm allocation (e.g. an A100 via `sintr`/`salloc`
# on the MPHIL-DIS-SL2-GPU account). Run this in the VS Code terminal on the node;
# VS Code auto-forwards the port to your laptop, so just open the forwarded URL.
#
#   bash hpc/run_interactive_server.sh           # real model (needs a GPU)
#   LLM_CIRCUITS_SERVE_MOCK=1 bash hpc/run_interactive_server.sh   # CPU mock (dev)
#
# Manual tunnel alternative (no VS Code):
#   ssh -L 8000:<gpu-node>:8000 <login>   then open http://localhost:8000
set -euo pipefail

REPO="$HOME/llm-circuits"
PATH="$HOME/.local/bin:$PATH"
PORT="${PORT:-8000}"

cd "$REPO"
set -a; source .env; set +a

# Bind localhost: VS Code forwards it; avoids exposing the model server on the network.
uv run --group serve uvicorn llm_circuits.serve.app:app --host 127.0.0.1 --port "$PORT"
