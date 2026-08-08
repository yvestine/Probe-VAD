#!/usr/bin/env bash
set -euo pipefail

# VideoLLaMA3 is already cached locally. Prevent each worker from contacting
# Hugging Face to check for repository updates at startup.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASET_DIR="${DATASET_DIR:-./data/ucf_crime}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${DATASET_DIR}/scores/videollama3_direct}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
GPU_IDS="${GPU_IDS:-2,3,6}"
FRAME_INTERVAL="${FRAME_INTERVAL:-16}"
WINDOW_SECONDS="${WINDOW_SECONDS:-10}"
SAMPLE_FPS="${SAMPLE_FPS:-2}"
MAX_FRAMES="${MAX_FRAMES:-10}"
SCORE_MODE="${SCORE_MODE:-generated}"
INITIAL_SCORE_MODE="${INITIAL_SCORE_MODE:-${SCORE_MODE}}"
REFINE_SCORE_MODE="${REFINE_SCORE_MODE:-${SCORE_MODE}}"
LIKELIHOOD_TEMPERATURE="${LIKELIHOOD_TEMPERATURE:-1.0}"
ALPHA="${ALPHA:-0.3}"
REBUILD_DERIVED="${REBUILD_DERIVED:-0}"
SAVE_FUSED="${SAVE_FUSED:-1}"
REFINEMENT_DIR_NAME="${REFINEMENT_DIR_NAME:-tag_refinement}"
REFINE_GATE="${REFINE_GATE:-0}"
GATE_MIN="${GATE_MIN:-0.45}"
GATE_MAX="${GATE_MAX:-0.55}"
REFINEMENT_ROOT="${EXPERIMENT_DIR}/${REFINEMENT_DIR_NAME}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_JOBS="${#GPUS[@]}"
mkdir -p "${EXPERIMENT_DIR}"

run_direct_job() {
  local gpu="$1"
  local job="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m src.video_direct_score \
    --video_dir "${VIDEO_DIR}" \
    --index_file "${INDEX_FILE}" \
    --output_dir "${EXPERIMENT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --device cuda:0 \
    --frame_interval "${FRAME_INTERVAL}" \
    --window_seconds "${WINDOW_SECONDS}" \
    --sample_fps "${SAMPLE_FPS}" \
    --max_frames "${MAX_FRAMES}" \
    --score_mode "${INITIAL_SCORE_MODE}" \
    --likelihood_temperature "${LIKELIHOOD_TEMPERATURE}" \
    --num_jobs "${NUM_JOBS}" \
    --job_index "${job}" \
    --resume
}

echo "[1/4] Generating direct scores (${INITIAL_SCORE_MODE}) on GPUs: ${GPU_IDS}"
for job in "${!GPUS[@]}"; do
  echo "Starting direct job ${job}/${NUM_JOBS} on physical GPU ${GPUS[$job]}"
  run_direct_job "${GPUS[$job]}" "${job}" &
done
wait

if [[ "${REBUILD_DERIVED}" == "1" ]]; then
  backup_dir="${EXPERIMENT_DIR}/derived_backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${backup_dir}"
  for derived_path in \
    "${EXPERIMENT_DIR}/highest_lowest_intervals.json" \
    "${EXPERIMENT_DIR}/suspicious_part_phrases.json" \
    "${REFINEMENT_ROOT}"; do
    if [[ -e "${derived_path}" ]]; then
      mv "${derived_path}" "${backup_dir}/"
    fi
  done
  echo "Moved old intervals, phrases, and refinement outputs to ${backup_dir}"
fi

echo "[2/4] Finding highest/lowest score windows"
python -m src.score_filter \
  --score_dir "${EXPERIMENT_DIR}" \
  --output "${EXPERIMENT_DIR}/highest_lowest_intervals.json"

echo "[3/4] Extracting suspicious activity phrases on physical GPU ${GPUS[0]}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -m src.summarize_window \
  --video_folder "${VIDEO_DIR}" \
  --index_file "${INDEX_FILE}" \
  --intervals_file "${EXPERIMENT_DIR}/highest_lowest_intervals.json" \
  --output_file "${EXPERIMENT_DIR}/suspicious_part_phrases.json" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --resume

run_refine_job() {
  local gpu="$1"
  local job="$2"
  local fused_flag
  local gate_flag
  if [[ "${SAVE_FUSED}" == "1" ]]; then
    fused_flag="--save_fused"
  else
    fused_flag="--no-save_fused"
  fi
  if [[ "${REFINE_GATE}" == "1" ]]; then
    gate_flag="--gate"
  else
    gate_flag="--no-gate"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python -m src.video_refine_with_tag \
    --video_dir "${VIDEO_DIR}" \
    --index_file "${INDEX_FILE}" \
    --initial_scores_dir "${EXPERIMENT_DIR}" \
    --phrases_file "${EXPERIMENT_DIR}/suspicious_part_phrases.json" \
    --output_root "${REFINEMENT_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --device cuda:0 \
    --alpha "${ALPHA}" \
    "${fused_flag}" \
    "${gate_flag}" \
    --gate_min "${GATE_MIN}" \
    --gate_max "${GATE_MAX}" \
    --frame_interval "${FRAME_INTERVAL}" \
    --window_seconds "${WINDOW_SECONDS}" \
    --sample_fps "${SAMPLE_FPS}" \
    --max_frames "${MAX_FRAMES}" \
    --score_mode "${REFINE_SCORE_MODE}" \
    --likelihood_temperature "${LIKELIHOOD_TEMPERATURE}" \
    --num_jobs "${NUM_JOBS}" \
    --job_index "${job}" \
    --resume
}

echo "[4/4] Generating tag-conditioned scores (${REFINE_SCORE_MODE}) on GPUs: ${GPU_IDS}"
if [[ "${REFINE_GATE}" == "1" ]]; then
  echo "Clip-level refinement gate: [${GATE_MIN}, ${GATE_MAX}]"
fi
for job in "${!GPUS[@]}"; do
  echo "Starting refine job ${job}/${NUM_JOBS} on physical GPU ${GPUS[$job]}"
  run_refine_job "${GPUS[$job]}" "${job}" &
done
wait

echo "Pipeline complete: ${EXPERIMENT_DIR}"
echo "Refinement output: ${REFINEMENT_ROOT}"
