#!/usr/bin/env bash
# gcp/setup_storage.sh — ONE-TIME: let the TPU (in PROJECT_ID) read/write the
# durable bucket (in GCS_PROJECT). Default = keyless: grant the dis project's
# default compute service account objectAdmin on YOUR bucket. This changes IAM
# on your bucket only, and asks before doing so.
#
# The bucket itself is created separately (already done for this project):
#   gcloud storage buckets create gs://$GCS_BUCKET --project=$GCS_PROJECT \
#       --location=us-east5 --uniform-bucket-level-access --public-access-prevention
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID GCS_PROJECT GCS_BUCKET

echo "Durable bucket: gs://$GCS_BUCKET  (project $GCS_PROJECT)"
gcloud storage buckets describe "gs://$GCS_BUCKET" --format="value(name,location)" \
    || { echo "Bucket not found — create it first (see the header of this script)."; exit 1; }

PROJ_NUM="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SA="${TPU_SA:-${PROJ_NUM}-compute@developer.gserviceaccount.com}"

echo
echo "TPU identity (default compute SA of $PROJECT_ID):"
echo "    $SA"
echo "Grant it roles/storage.objectAdmin on gs://$GCS_BUCKET (keyless cross-project access)."
confirm "Proceed with this IAM grant?" || { echo "aborted"; exit 0; }

gcloud storage buckets add-iam-policy-binding "gs://$GCS_BUCKET" \
    --member="serviceAccount:$SA" \
    --role="roles/storage.objectAdmin"

echo
echo "Done. After provisioning a TPU, verify access from the VM:"
echo "    gcp/ssh.sh 'echo ok | gcloud storage cp - gs://$GCS_BUCKET/_auth_test && gcloud storage cat gs://$GCS_BUCKET/_auth_test'"
echo
echo "FALLBACK — if the VM's default SA is disabled or lacks scope, use an SA key:"
echo "    gcloud iam service-accounts create tpu-storage --project=$GCS_PROJECT"
echo "    gcloud storage buckets add-iam-policy-binding gs://$GCS_BUCKET \\"
echo "        --member=serviceAccount:tpu-storage@$GCS_PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin"
echo "    gcloud iam service-accounts keys create gcp/keys/sa-key.json \\"
echo "        --iam-account=tpu-storage@$GCS_PROJECT.iam.gserviceaccount.com   # gitignored"
echo "    # scp gcp/keys/sa-key.json to the VM and set GOOGLE_APPLICATION_CREDENTIALS in the VM .env"
