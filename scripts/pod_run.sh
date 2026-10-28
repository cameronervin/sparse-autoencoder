#!/usr/bin/env bash
# The GPU part of the pipeline, run on the pod inside tmux:
#   tmux new -s sae 'bash scripts/pod_run.sh configs/full.yaml 2>&1 | tee artifacts/pod_run.log'
# Labeling runs on the laptop afterwards, so the API key never goes to the pod.
set -euo pipefail
cfg=${1:-configs/full.yaml}
mkdir -p artifacts
uv run gpt2-sae train --config "$cfg" --resume
uv run gpt2-sae evaluate --config "$cfg" --compare
uv run gpt2-sae features --config "$cfg"
echo "done: copy artifacts/ and results/ back with scripts/pod_fetch.sh"
