#!/usr/bin/env bash
# gcp/status.sh — show queued-resource state, TPU state, and bucket contents.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE

echo "=== queued resource: ${QR_NAME:-?} ==="
gcloud compute tpus queued-resources describe "${QR_NAME:-}" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --format="value(state.state)" 2>/dev/null || echo "(none)"

echo "=== tpu vm: ${TPU_NAME:-?} ==="
gcloud compute tpus tpu-vm describe "${TPU_NAME:-}" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --format="value(state,acceleratorType,runtimeVersion)" 2>/dev/null || echo "(none)"

echo "=== storage ==="
if use_bucket; then
    gcloud storage ls "gs://$GCS_BUCKET/" 2>/dev/null || echo "(bucket empty or unreachable)"
else
    echo "local-pull mode (GCS_BUCKET empty)"
fi
