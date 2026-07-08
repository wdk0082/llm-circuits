#!/usr/bin/env bash
# gcp/launch.sh <script.py> [args...] — run an experiment on the TPU.
#
# Pulls the latest committed code onto the VM, then runs the script through
# ./bin/run with LLM_CIRCUITS_DEVICE=tpu and the artifact/checkpoint dirs
# injected as caller-overrides (they win over the VM .env). Output streams to
# your terminal. Artifacts are staged to on-VM scratch and, in bucket mode,
# synced to GCS after the run; in local-pull mode they are scp'd back.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE TPU_NAME GIT_REMOTE
[[ $# -ge 1 ]] || { echo "usage: gcp/launch.sh examples/<script>.py [args...]" >&2; exit 1; }

REPO_NAME="$(basename "${GIT_REMOTE%.git}")"
REF="${GIT_REF:-main}"

# Sync the VM to the latest committed code (commit + push before launching).
tpu_ssh --command "cd \$HOME/$REPO_NAME && git fetch origin $REF && git checkout $REF && git pull --ff-only"

# LLM_CIRCUITS_DEVICE=tpu marks the run for the TPU (torch_xla). Artifacts
# stage to on-VM scratch; $HOME expands on the VM (see the escaped \$HOME).
inject="LLM_CIRCUITS_DEVICE=tpu LLM_CIRCUITS_ARTIFACTS_DIR=\$HOME/scratch/artifacts"
if use_bucket; then
    inject="$inject CKPT_DIR='gs://$GCS_BUCKET/checkpoints' GCS_ARTIFACTS='gs://$GCS_BUCKET/artifacts'"
fi

# Forward experiment knobs set inline (e.g. `DRY_RUN=1 gcp/launch.sh ...`);
# ssh does not carry the caller's env, so we splice any that are set into the
# command. Add per-project experiment knobs to this list as they appear.
# WANDB_* come from the laptop .env (sourced by lib.sh) so on-TPU training
# authenticates and logs online to your wandb account; the API key is spliced
# into the remote command, which is fine for an ephemeral personal VM.
for v in DRY_RUN HF_TOKEN \
         WANDB_API_KEY WANDB_MODE WANDB_PROJECT WANDB_ENTITY; do
    [[ -n "${!v:-}" ]] && inject="$inject $v='${!v}'"
done

echo "Launching on $TPU_NAME:  $*"
tpu_ssh --command "cd \$HOME/$REPO_NAME && PYTHONUNBUFFERED=1 $inject ./bin/run python -u $*"

# Bucket mode: sync the staged artifacts to GCS so they survive VM deletion.
# Local-pull mode: copy them back to the laptop now, before the VM is gone.
if use_bucket; then
    echo "Syncing artifacts -> gs://$GCS_BUCKET/artifacts …"
    tpu_ssh --command "gcloud storage rsync --recursive \$HOME/scratch/artifacts gs://$GCS_BUCKET/artifacts" || true
else
    echo "Local-pull mode: copying artifacts back…"
    "$GCP_DIR/pull.sh" || true
fi
