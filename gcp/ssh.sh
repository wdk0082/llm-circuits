#!/usr/bin/env bash
# gcp/ssh.sh [cmd...] — open an interactive shell on the TPU, or run a command.
#   gcp/ssh.sh                         # interactive shell
#   gcp/ssh.sh 'nvidia-smi || ls ~'    # run a one-off command
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE TPU_NAME

if [[ $# -gt 0 ]]; then
    tpu_ssh --command "$*"
else
    tpu_ssh
fi
