#!/usr/bin/env bash
# Idempotent pod setup. uv, Python, the venv and the Hugging Face cache all live on the
# /workspace volume, so a new pod on the same volume reuses them. Then: source /workspace/env.sh
set -euo pipefail
ws=/workspace

cat > "$ws/env.sh" <<ENV
# RunPod puts pod env vars (including secrets) in this file; SSH sessions don't inherit them.
[[ -f /etc/rp_environment ]] && source /etc/rp_environment
# The image turns on hf_transfer, which is not in this project's venv.
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH=$ws/.local/bin:\$PATH
export UV_CACHE_DIR=$ws/.uv-cache UV_PYTHON_INSTALL_DIR=$ws/.uv-python HF_HOME=$ws/hf
# nproc reports the host's CPUs, not the pod's quota.
export OMP_NUM_THREADS=\${RUNPOD_CPU_COUNT:-\$(nproc)} RAYON_NUM_THREADS=\${RUNPOD_CPU_COUNT:-\$(nproc)}
cd $ws/sparse-autoencoder
ENV
# shellcheck disable=SC1091
source "$ws/env.sh"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$ws/.local/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv sync --frozen --extra compare
command -v tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux)

echo "gpu:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "vcpus:  ${RUNPOD_CPU_COUNT:-$(nproc)} (quota)"
echo "volume: $(df -h "$ws" | tail -1)"
