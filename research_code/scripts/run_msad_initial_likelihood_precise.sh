#!/usr/bin/env bash
set -euo pipefail

# Resume the MSAD direct 11-class likelihood run with exact timestamp sampling.
# Existing valid clips are preserved; clips listed in _errors are retried.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PYTHONPATH:-.}"
stty -tostop 2>/dev/null || true

python -m src.patch_videollama3_timestamp

OUT="${OUTPUT_DIR:-./data/MSAD/scores/videollama3_direct_11class_likelihood_optimized_stride16_promptfix}"
INDEX_FILE="${INDEX_FILE:-./results/msad/annotations/test.txt}"
COMMON=(
  --video_dir ./data/MSAD/videos
  --index_file "$INDEX_FILE"
  --output_dir "$OUT"
  --model_path DAMO-NLP-SG/VideoLLaMA3-7B
  --device cuda:0
  --frame_interval 16
  --window_seconds 10
  --sample_fps 2
  --max_frames 10
  --score_mode likelihood_optimized
  --likelihood_temperature 1.0
  --length_normalize
  --precise_time
  --num_jobs 2
  --resume
)

pkill -TERM -u "$(id -u)" \
  -f '[v]ideo_direct_score.*MSAD' || true
sleep 5
pkill -KILL -u "$(id -u)" \
  -f '[v]ideo_direct_score.*MSAD' || true

CUDA_VISIBLE_DEVICES=2 python -u -m src.video_direct_score \
  "${COMMON[@]}" --job_index 0 &
P0=$!

CUDA_VISIBLE_DEVICES=3 python -u -m src.video_direct_score \
  "${COMMON[@]}" --job_index 1 &
P1=$!

wait "$P0" "$P1"
