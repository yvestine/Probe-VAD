#!/usr/bin/env bash
set -euo pipefail

# Queue the MSAD E4 center-2s ablation after the center-4s experiment.
# This script is safe in either situation:
#   1. center-4s is still running: wait for its final metric;
#   2. center-4s was stopped: resume it from its score JSON checkpoints.

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GPU_IDS="${GPU_IDS:-3,4,6}"
DATASET_DIR="${DATASET_DIR:-./data/MSAD}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
INDEX_FILE="${INDEX_FILE:-./VADTree/dataset_info/MSAD/annotations/anomaly_test.txt}"
EVAL_ANNOTATION_FILE="${EVAL_ANNOTATION_FILE:-${INDEX_FILE}}"
TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE:-./VADTree/dataset_info/MSAD/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt}"
ROOT_PATH="${ROOT_PATH:-${DATASET_DIR}/frames}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"

CENTER4_OUTPUT_DIR="${CENTER4_OUTPUT_DIR:-${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e4_center_dense_g4_c6_s4}"
CENTER2_OUTPUT_DIR="${CENTER2_OUTPUT_DIR:-${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e4_center_dense_g4_c6_s2}"
CENTER4_METRIC="${CENTER4_OUTPUT_DIR}/metrics/roc_auc.txt"

FRAME_INTERVAL="${FRAME_INTERVAL:-16}"
WINDOW_SECONDS="${WINDOW_SECONDS:-10}"
SAMPLE_FPS="${SAMPLE_FPS:-2}"
MAX_FRAMES="${MAX_FRAMES:-10}"
GLOBAL_FRAMES="${GLOBAL_FRAMES:-4}"
CENTER_FRAMES="${CENTER_FRAMES:-6}"
DECODE_MAX_FRAMES="${DECODE_MAX_FRAMES:-64}"
PRECISE_TIME="${PRECISE_TIME:-1}"
LIKELIHOOD_TEMPERATURE="${LIKELIHOOD_TEMPERATURE:-1.0}"
NORMAL_LABEL="${NORMAL_LABEL:-0}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-10}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

if [[ "${CENTER4_OUTPUT_DIR}" == "${CENTER2_OUTPUT_DIR}" ]]; then
  echo "CENTER4_OUTPUT_DIR and CENTER2_OUTPUT_DIR must differ." >&2
  exit 2
fi
if ((WAIT_SECONDS <= 0)); then
  echo "WAIT_SECONDS must be positive." >&2
  exit 2
fi

run_msad_e4() {
  local center_seconds="$1"
  local output_dir="$2"
  echo
  echo "Starting/resuming MSAD E4 with CENTER_SECONDS=${center_seconds}"
  echo "GPUs: ${GPU_IDS}"
  echo "Output: ${output_dir}"
  GPU_IDS="${GPU_IDS}" \
  DATASET_DIR="${DATASET_DIR}" \
  VIDEO_DIR="${VIDEO_DIR}" \
  INDEX_FILE="${INDEX_FILE}" \
  EVAL_ANNOTATION_FILE="${EVAL_ANNOTATION_FILE}" \
  TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE}" \
  ROOT_PATH="${ROOT_PATH}" \
  OUTPUT_DIR="${output_dir}" \
  MODEL_PATH="${MODEL_PATH}" \
  NORMAL_LABEL="${NORMAL_LABEL}" \
  FRAME_INTERVAL="${FRAME_INTERVAL}" \
  WINDOW_SECONDS="${WINDOW_SECONDS}" \
  SAMPLE_FPS="${SAMPLE_FPS}" \
  MAX_FRAMES="${MAX_FRAMES}" \
  GLOBAL_FRAMES="${GLOBAL_FRAMES}" \
  CENTER_FRAMES="${CENTER_FRAMES}" \
  CENTER_SECONDS="${center_seconds}" \
  DECODE_MAX_FRAMES="${DECODE_MAX_FRAMES}" \
  PRECISE_TIME="${PRECISE_TIME}" \
  LIKELIHOOD_TEMPERATURE="${LIKELIHOOD_TEMPERATURE}" \
  THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE}" \
  PREFIX_CACHE="${PREFIX_CACHE}" \
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
  bash scripts/run_video_center_dense_rebalanced.sh
}

center4_worker_running() {
  pgrep -f \
    "[s]rc.video_center_dense_score.*--output_dir ${CENTER4_OUTPUT_DIR}" \
    >/dev/null
}

if [[ -s "${CENTER4_METRIC}" ]]; then
  echo "CENTER_SECONDS=4 is already complete: ${CENTER4_METRIC}"
elif center4_worker_running; then
  echo "CENTER_SECONDS=4 is currently running; waiting without interrupting it."
  while [[ ! -s "${CENTER4_METRIC}" ]]; do
    if ! center4_worker_running; then
      echo "Center-4s workers are no longer visible; waiting one interval for evaluation."
      sleep "${WAIT_SECONDS}"
      if [[ ! -s "${CENTER4_METRIC}" ]]; then
        echo "No center-4s metric found; resuming from saved checkpoints."
        run_msad_e4 4 "${CENTER4_OUTPUT_DIR}"
      fi
      break
    fi
    echo "$(date '+%F %T') center-4s is still running; next check in ${WAIT_SECONDS}s"
    sleep "${WAIT_SECONDS}"
  done
else
  echo "CENTER_SECONDS=4 is incomplete and not running; resuming checkpoints."
  run_msad_e4 4 "${CENTER4_OUTPUT_DIR}"
fi

if [[ ! -s "${CENTER4_METRIC}" ]]; then
  echo "Center-4s did not produce ${CENTER4_METRIC}; refusing to start center-2s." >&2
  exit 1
fi

echo
echo "CENTER_SECONDS=4 completed."
printf 'Center-4s ROC-AUC: %s\n' "$(tr -d '[:space:]' < "${CENTER4_METRIC}")"

run_msad_e4 2 "${CENTER2_OUTPUT_DIR}"

echo
echo "MSAD E4 center-duration sequence completed."
echo "Center-4s results: ${CENTER4_OUTPUT_DIR}"
echo "Center-2s results: ${CENTER2_OUTPUT_DIR}"
