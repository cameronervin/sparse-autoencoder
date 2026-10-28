#!/usr/bin/env bash
# Copy run outputs from the pod back to this checkout. Skips the optimizer checkpoint.
#   scripts/pod_fetch.sh sae-pod [run-name]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
host=${1:?usage: scripts/pod_fetch.sh <ssh-host> [run-name]}
name=${2:-full}
remote=/workspace/sparse-autoencoder
mkdir -p "artifacts/$name" "results/$name"
rsync -avz --exclude ckpt.pt "$host:$remote/artifacts/$name/" "artifacts/$name/"
rsync -avz "$host:$remote/results/$name/" "results/$name/"
