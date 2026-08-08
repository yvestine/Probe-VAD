#!/usr/bin/env bash
set -euo pipefail

# Pure direct 11-class ordinal likelihood ablation.
# This script intentionally does not run score_filter, summarize_window,
# caption generation, tag refinement, or score fusion.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASET_DIR="${DATASET_DIR:-./data/ucf_crime}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
EVAL_ANNOTATION_FILE="${EVAL_ANNOTATION_FILE:-${INDEX_FILE}}"
TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE:-${DATASET_DIR}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt}"
ROOT_PATH="${ROOT_PATH:-${DATASET_DIR}/frames}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_stride32_interp16}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
GPU_IDS="${GPU_IDS:-0,1}"
FRAME_INTERVAL="${FRAME_INTERVAL:-32}"
OUTPUT_INTERVAL="${OUTPUT_INTERVAL:-16}"
WINDOW_SECONDS="${WINDOW_SECONDS:-10}"
SAMPLE_FPS="${SAMPLE_FPS:-2}"
MAX_FRAMES="${MAX_FRAMES:-10}"
NORMAL_LABEL="${NORMAL_LABEL:-7}"
LIKELIHOOD_TEMPERATURE="${LIKELIHOOD_TEMPERATURE:-1.0}"
LENGTH_NORMALIZE="${LENGTH_NORMALIZE:-1}"
SCORE_MODE="${SCORE_MODE:-likelihood}"

# Background workers share the tmux terminal.  Without this, terminals with
# tostop enabled suspend workers as soon as tqdm/log output is written.
stty -tostop 2>/dev/null || true

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_JOBS="${#GPUS[@]}"
mkdir -p "${OUTPUT_DIR}"

length_flag="--length_normalize"
if [[ "${LENGTH_NORMALIZE}" == "0" ]]; then
  length_flag="--no-length_normalize"
fi

echo "Direct 11-class likelihood scoring"
echo "GPUs: ${GPU_IDS}; jobs: ${NUM_JOBS}; output: ${OUTPUT_DIR}"

pids=()
for job in "${!GPUS[@]}"; do
  echo "Starting job ${job}/${NUM_JOBS} on physical GPU ${GPUS[$job]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$job]}" python -u -m src.video_direct_score \
    --video_dir "${VIDEO_DIR}" \
    --index_file "${INDEX_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --device cuda:0 \
    --frame_interval "${FRAME_INTERVAL}" \
    --window_seconds "${WINDOW_SECONDS}" \
    --sample_fps "${SAMPLE_FPS}" \
    --max_frames "${MAX_FRAMES}" \
    --score_mode "${SCORE_MODE}" \
    --likelihood_temperature "${LIKELIHOOD_TEMPERATURE}" \
    "${length_flag}" \
    --num_jobs "${NUM_JOBS}" \
    --job_index "${job}" \
    --resume &
  pids+=("$!")
done

wait "${pids[@]}"

echo "Evaluating direct 11-class likelihood scores"
python -m src.eval_interpolated \
  --root_path "${ROOT_PATH}" \
  --annotationfile_path "${EVAL_ANNOTATION_FILE}" \
  --temporal_annotation_file "${TEMPORAL_ANNOTATION_FILE}" \
  --scores_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/metrics" \
  --scoring_interval "${FRAME_INTERVAL}" \
  --output_interval "${OUTPUT_INTERVAL}" \
  --normal_label "${NORMAL_LABEL}"

echo "Direct 11-class likelihood complete: ${OUTPUT_DIR}"
