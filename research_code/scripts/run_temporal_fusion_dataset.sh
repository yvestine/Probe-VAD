#!/usr/bin/env bash
set -euo pipefail

# CPU-only temporal fusion for one dataset.  It reuses the completed E0
# window scores and only recomputes temporal aggregation plus evaluation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/gujiawei/miniconda3/envs/VAA/bin/python}"
DATASET_DIR="${1:?dataset directory}"
INDEX_FILE="${2:?index file}"
TEMPORAL_FILE="${3:?temporal annotation file}"
NORMAL_LABEL="${4:?normal label}"
DATASET_NAME="${5:?dataset name}"

E0="${DATASET_DIR}/scores/videollama3_cumulative_likelihood_optimized"
[[ -d "${E0}" ]] || { echo "E0 output missing: ${E0}" >&2; exit 2; }

for mode in overlap_mean overlap_logit adaptive; do
  output="${DATASET_DIR}/scores/videollama3_e0_temporal_fusion_${mode}"
  echo "[${DATASET_NAME}] temporal fusion=${mode}"
  "${PYTHON_BIN}" -u -m src.temporal_window_fusion \
    --scores_dir "${E0}" --video_dir "${DATASET_DIR}/videos" --index_file "${INDEX_FILE}" \
    --output_dir "${output}" --fusion_mode "${mode}" --window_seconds 10 \
    --no-save_details --log_level WARNING --resume
  "${PYTHON_BIN}" -u -m src.eval --root_path "${DATASET_DIR}/frames" \
    --annotationfile_path "${INDEX_FILE}" --temporal_annotation_file "${TEMPORAL_FILE}" \
    --scores_dir "${output}" --output_dir "${output}/metrics_sigma10" \
    --frame_interval 16 --normal_label "${NORMAL_LABEL}" --smoothing_sigma 10
  printf '[%s/%s] ROC=%s PR=%s MaxF1=%s\n' "${DATASET_NAME}" "${mode}" \
    "$(tr -d '[:space:]' < "${output}/metrics_sigma10/roc_auc.txt")" \
    "$(tr -d '[:space:]' < "${output}/metrics_sigma10/pr_auc.txt")" \
    "$(tr -d '[:space:]' < "${output}/metrics_sigma10/max_f1.txt")"
done
