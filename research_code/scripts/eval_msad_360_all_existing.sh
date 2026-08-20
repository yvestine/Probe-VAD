#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate every existing MSAD score directory on the complete 360-video
# protocol. This is CPU-only and never deletes or overwrites the historical
# metrics directories (240-video results remain in place).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASET_DIR="${DATASET_DIR:-./data/MSAD}"
SCORES_ROOT="${SCORES_ROOT:-${DATASET_DIR}/scores}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
TEMPORAL_FILE="${TEMPORAL_FILE:-${DATASET_DIR}/annotations/msad_anomaly_index.txt}"
ROOT_PATH="${ROOT_PATH:-${DATASET_DIR}/frames}"
NORMAL_LABEL="${NORMAL_LABEL:-0}"
FRAME_INTERVAL="${FRAME_INTERVAL:-16}"

[[ "$(wc -l < "${INDEX_FILE}")" -eq 360 ]] || {
  echo "Expected the complete 360-video MSAD index: ${INDEX_FILE}" >&2
  exit 2
}
[[ "$(wc -l < "${TEMPORAL_FILE}")" -eq 360 ]] || {
  echo "Expected the complete 360-video temporal annotation: ${TEMPORAL_FILE}" >&2
  exit 2
}

printf '%s\n' "MSAD 360 evaluation (no GPU inference)"
printf '%-62s %8s %8s %8s %8s\n' "score directory" "videos" "ROC" "PR" "Max-F1"
for score_dir in "${SCORES_ROOT}"/*; do
  [[ -d "${score_dir}" ]] || continue
  [[ "$(basename "${score_dir}")" == _* ]] && continue
  json_count="$(find "${score_dir}" -maxdepth 1 -type f -name '*.json' | wc -l)"
  (( json_count > 0 )) || continue
  if (( json_count < 360 )); then
    printf '%-62s %8d %s\n' "$(basename "${score_dir}")" "${json_count}" "SKIP: incomplete; resume inference first"
    continue
  fi
  metrics_dir="${score_dir}/metrics_360"
  if [[ "$(basename "${score_dir}")" == *stride32* ]]; then
    python -u -m src.eval_interpolated \
      --root_path "${ROOT_PATH}" \
      --annotationfile_path "${INDEX_FILE}" \
      --temporal_annotation_file "${TEMPORAL_FILE}" \
      --scores_dir "${score_dir}" \
      --output_dir "${metrics_dir}" \
      --scoring_interval 32 --output_interval 16 \
      --normal_label "${NORMAL_LABEL}"
  else
    python -u -m src.eval \
      --root_path "${ROOT_PATH}" \
      --annotationfile_path "${INDEX_FILE}" \
      --temporal_annotation_file "${TEMPORAL_FILE}" \
      --scores_dir "${score_dir}" \
      --output_dir "${metrics_dir}" \
      --frame_interval "${FRAME_INTERVAL}" \
      --normal_label "${NORMAL_LABEL}" \
      --smoothing_sigma 10
  fi
  roc="$(tr -d '[:space:]' < "${metrics_dir}/roc_auc.txt")"
  pr="$(tr -d '[:space:]' < "${metrics_dir}/pr_auc.txt")"
  f1="$(tr -d '[:space:]' < "${metrics_dir}/max_f1.txt")"
  printf '%-62s %8d %8s %8s %8s\n' "$(basename "${score_dir}")" "${json_count}" "${roc}" "${pr}" "${f1}"
done
