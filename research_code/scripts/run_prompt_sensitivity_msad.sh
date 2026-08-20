#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib_dataset_paths.sh"

# MSAD prompt sensitivity: one model process and one visual encoding per
# window, followed by four prompt variants x ten threshold questions.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
GPU_IDS="${GPU_IDS:-0}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/configs/covas_prompt_suite.json}"
MSAD_DIR="${ROOT_DIR}/data/MSAD"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MSAD_DIR}/scores/covas_prompt_sensitivity}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-40}"

if [[ "${GPU_IDS}" == *,* ]]; then
  echo "This combined prompt launcher uses one GPU; set GPU_IDS to one physical GPU." >&2
  exit 2
fi

INDEX_FILE="${ROOT_DIR}/data/MSAD/annotations/test.txt"
TEMPORAL_FILE="${ROOT_DIR}/data/MSAD/annotations/msad_anomaly_index.txt"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

python -u -m src.video_prompt_sensitivity \
  --video_dir "${MSAD_DIR}/videos" \
  --index_file "${INDEX_FILE}" \
  --output_root "${OUTPUT_ROOT}" \
  --prompt_file "${PROMPT_FILE}" \
  --prompt_variants visible_evidence reach_level no_less_than rated_above \
  --model_path "${MODEL_PATH}" \
  --backend videollama3 \
  --device cuda:0 \
  --frame_interval 16 \
  --window_seconds 10 \
  --sample_fps 2 \
  --max_frames 10 \
  --no-precise_time \
  --likelihood_temperature 1.0 \
  --optimized \
  --threshold_batch_size "${THRESHOLD_BATCH_SIZE}" \
  --prefix_cache \
  --checkpoint_interval 20 \
  --no-save_threshold_details \
  --monotonic_projection \
  --resume

for prompt_id in visible_evidence reach_level no_less_than rated_above; do
  output_dir="${OUTPUT_ROOT}/${prompt_id}"
  mkdir -p "${output_dir}/metrics"
  python -m src.eval \
    --root_path "${MSAD_DIR}/frames" \
    --annotationfile_path "${INDEX_FILE}" \
    --temporal_annotation_file "${TEMPORAL_FILE}" \
    --scores_dir "${output_dir}" \
    --output_dir "${output_dir}/metrics_360" \
    --frame_interval 16 \
    --normal_label 7
  echo "${prompt_id}: ROC=$(tr -d '[:space:]' < "${output_dir}/metrics_360/roc_auc.txt") PR=$(tr -d '[:space:]' < "${output_dir}/metrics_360/pr_auc.txt") MaxF1=$(tr -d '[:space:]' < "${output_dir}/metrics_360/max_f1.txt")"
done
