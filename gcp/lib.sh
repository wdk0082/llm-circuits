#!/usr/bin/env bash
# gcp/lib.sh — shared helpers for the Cloud TPU lifecycle scripts.
# Source this at the top of every gcp/ script: it loads .env and defines
# require / confirm / tpu_ssh / use_bucket. These scripts run on your LAPTOP
# and drive the TPU via gcloud; the TPU itself runs `./bin/run` (see CLAUDE.md).
set -euo pipefail

GCP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$GCP_DIR/.." && pwd)"

if [[ ! -f "$REPO_DIR/.env" ]]; then
    echo "gcp: missing $REPO_DIR/.env (copy .env.example -> .env)" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source "$REPO_DIR/.env"
set +a

WORKER="${WORKER:-0}"

# require VAR...  — abort if any named var is unset/empty
require() {
    local missing=0 v
    for v in "$@"; do
        if [[ -z "${!v:-}" ]]; then
            echo "gcp: required env var '$v' is empty — set it in .env" >&2
            missing=1
        fi
    done
    [[ "$missing" -eq 0 ]] || exit 1
}

# confirm "message"  — y/N prompt; auto-yes when FORCE=1
confirm() {
    [[ "${FORCE:-0}" == "1" ]] && return 0
    local reply
    read -r -p "$1 [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

# tpu_ssh [gcloud-ssh-args...]  — ssh to TPU worker $WORKER
tpu_ssh() {
    require PROJECT_ID ZONE TPU_NAME
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --project="$PROJECT_ID" --zone="$ZONE" --worker="$WORKER" "$@"
}

# use_bucket  — true when a durable GCS bucket is configured (vs local-pull mode)
use_bucket() { [[ -n "${GCS_BUCKET:-}" ]]; }
