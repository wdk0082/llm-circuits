#!/usr/bin/env bash
# gcp/teardown.sh — delete the TPU and its queued resource. Durable data in
# GCS is untouched. In local-pull mode, artifacts are pulled first so the
# VM's local outputs aren't lost.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE QR_NAME

if ! use_bucket; then
    echo "Local-pull mode: pulling artifacts before delete…"
    "$GCP_DIR/pull.sh" || echo "  WARNING: pull failed — inspect the VM before deleting!"
fi

confirm "Delete queued resource '$QR_NAME' and TPU '${TPU_NAME:-?}' in $ZONE?" \
    || { echo "aborted"; exit 0; }

# --force also deletes the underlying node if it's still ACTIVE.
gcloud compute tpus queued-resources delete "$QR_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" --force --quiet

echo "Deleted. Durable data remains in gs://${GCS_BUCKET:-<none — local-pull mode>}."
