#!/usr/bin/env bash
# gcp/create.sh — provision the TPU as a queued resource, then bootstrap it.
# Spot by default (TPU_SPOT=1, cheap + preemptible); set TPU_SPOT=0 for on-demand.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE TPU_NAME QR_NAME ACCELERATOR_TYPE RUNTIME_VERSION

spot_flag=""
[[ "${TPU_SPOT:-1}" == "1" ]] && spot_flag="--spot"

echo "Creating queued resource '$QR_NAME' -> node '$TPU_NAME'"
echo "  $ACCELERATOR_TYPE / $RUNTIME_VERSION in $ZONE  (${spot_flag:-on-demand})"

gcloud compute tpus queued-resources create "$QR_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --node-id="$TPU_NAME" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --runtime-version="$RUNTIME_VERSION" \
    $spot_flag

echo "Waiting for ACTIVE (queued resources can wait for capacity; Ctrl-C is safe)…"
until [[ "$(gcloud compute tpus queued-resources describe "$QR_NAME" \
            --project="$PROJECT_ID" --zone="$ZONE" \
            --format='value(state.state)' 2>/dev/null)" == "ACTIVE" ]]; do
    sleep 20
    printf '.'
done
echo " ACTIVE"

echo "Bootstrapping the VM…"
"$GCP_DIR/bootstrap.sh"
