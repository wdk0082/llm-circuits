#!/usr/bin/env bash
# gcp/pull.sh — bring artifacts to the laptop for inspection / committing.
#   bucket mode : rsync gs://<bucket>/artifacts -> $LOCAL_ARTIFACTS
#   local-pull  : scp $HOME/scratch/artifacts from the VM -> $LOCAL_ARTIFACTS
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dest="${LOCAL_ARTIFACTS:-./artifacts}"
mkdir -p "$dest"

if use_bucket; then
    echo "Syncing gs://$GCS_BUCKET/artifacts -> $dest"
    gcloud storage rsync --recursive "gs://$GCS_BUCKET/artifacts" "$dest"
else
    require PROJECT_ID ZONE TPU_NAME
    echo "Copying VM artifacts -> $dest"
    gcloud compute tpus tpu-vm scp --recurse \
        "$TPU_NAME:~/scratch/artifacts" "$dest" \
        --project="$PROJECT_ID" --zone="$ZONE" --worker="$WORKER"
fi
echo "Pulled to $dest"
