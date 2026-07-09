#!/usr/bin/env bash
# gcp/pull.sh — bring artifacts to the laptop for inspection / committing.
#   bucket mode : rsync $GCS_ARTIFACTS (from .env) -> $LOCAL_ARTIFACTS
#   local-pull  : scp $HOME/scratch/artifacts from the VM -> $LOCAL_ARTIFACTS
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dest="${LOCAL_ARTIFACTS:-./artifacts}"
mkdir -p "$dest"

if use_bucket; then
    src="${GCS_ARTIFACTS:-gs://$GCS_BUCKET/artifacts}"
    echo "Syncing $src -> $dest"
    gcloud storage rsync --recursive "$src" "$dest"
else
    require PROJECT_ID ZONE TPU_NAME
    echo "Copying VM artifacts -> $dest"
    gcloud compute tpus tpu-vm scp --recurse \
        "$TPU_NAME:~/scratch/artifacts" "$dest" \
        --project="$PROJECT_ID" --zone="$ZONE" --worker="$WORKER"
fi
echo "Pulled to $dest"
