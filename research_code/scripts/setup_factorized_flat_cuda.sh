#!/usr/bin/env bash
# Execute on Linux NVIDIA server; creates a NEW env, never alters an existing env.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
[[ "$(uname -s)" == Linux ]] || { echo 'Run this installer on Linux/NVIDIA.' >&2; exit 2; }
command -v conda >/dev/null || { echo 'Install Miniconda/Conda first.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/nvidia-smi required.' >&2; exit 2; }
ENV_NAME="${ENV_NAME:-probe-vad-factorized}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
[[ "$ATTN_IMPLEMENTATION" == flash_attention_2 || "$ATTN_IMPLEMENTATION" == eager ]] || exit 2
# conda env create refuses an existing prefix, protecting the original VAA environment.
conda env create --name "$ENV_NAME" --file environment.yml
conda install -y -n "$ENV_NAME" -c conda-forge ffmpeg
conda run -n "$ENV_NAME" python -m pip install -e . --no-deps
conda run -n "$ENV_NAME" python -m pip install pytest==8.3.5 packaging ninja einops ffmpeg-python imageio
if [[ "$ATTN_IMPLEMENTATION" == flash_attention_2 ]]; then
  command -v nvcc >/dev/null || {
    echo 'FlashAttention build needs a compatible CUDA 12.4 toolkit/nvcc and C++ compiler.' >&2
    echo "Environment already created. Install compatible flash-attn in $ENV_NAME, or run with ATTN_IMPLEMENTATION=eager." >&2
    exit 2
  }
  conda run -n "$ENV_NAME" python -m pip install flash-attn --no-build-isolation --no-deps
fi
conda run -n "$ENV_NAME" python -m pip check
conda run -n "$ENV_NAME" python -m pytest tests/test_factorized_flat.py -q
PYTHONPATH="$REPO/research_code${PYTHONPATH:+:$PYTHONPATH}" \
  conda run -n "$ENV_NAME" python -m src.factorized_cuda_preflight --attn_implementation "$ATTN_IMPLEMENTATION"
echo "Ready: conda activate $ENV_NAME"
