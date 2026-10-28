#!/usr/bin/env bash
# Copy this repo to a RunPod pod's /workspace volume and set it up there.
# Uses a git bundle over SSH, so a private repo needs no extra credentials.
#
#   scripts/pod_sync.sh sae-pod        # an SSH host alias from ~/.ssh/config
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

host=${1:?usage: scripts/pod_sync.sh <ssh-host>}
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
  echo "warning: uncommitted changes are not synced; commit them first" >&2
fi

git bundle create - --all | ssh "$host" 'cat > /workspace/sae.bundle'
ssh "$host" bash -s -- "$branch" <<'REMOTE'
set -euo pipefail
branch=$1
repo=/workspace/sparse-autoencoder
if [[ -d "$repo/.git" ]]; then
  git -C "$repo" fetch -q /workspace/sae.bundle "+refs/heads/*:refs/remotes/bundle/*"
  git -C "$repo" reset -q --hard
  git -C "$repo" checkout -q -B "$branch" "bundle/$branch"
else
  git clone -q -b "$branch" /workspace/sae.bundle "$repo"
fi
rm -f /workspace/sae.bundle
echo "synced: $(git -C "$repo" log --oneline -1)"
bash "$repo/scripts/pod_setup.sh"
REMOTE
