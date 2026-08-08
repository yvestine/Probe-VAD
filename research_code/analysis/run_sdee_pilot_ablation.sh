#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-./data/ucf_crime}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
FULL_INDEX_FILE="${FULL_INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
E0_SCORES_DIR="${E0_SCORES_DIR:-${DATASET_DIR}/scores/videollama3_cumulative_likelihood}"
ROOT_PATH="${ROOT_PATH:-${DATASET_DIR}/frames}"
TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE:-${DATASET_DIR}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt}"
NORMAL_LABEL="${NORMAL_LABEL:-7}"
GPU_IDS="${GPU_IDS:-0}"
PILOT_VIDEOS="${PILOT_VIDEOS:-30}"
PILOT_SEED="${PILOT_SEED:-2026}"
PILOT_ROOT="${PILOT_ROOT:-${DATASET_DIR}/experiments/e5_sdee_pilot_${PILOT_VIDEOS}}"
PILOT_INDEX_FILE="${PILOT_INDEX_FILE:-${PILOT_ROOT}/pilot_index.txt}"
PRECISE_TIME="${PRECISE_TIME:-0}"
MOTION_FPS="${MOTION_FPS:-8}"
SAVE_MOTION_DETAILS="${SAVE_MOTION_DETAILS:-1}"

mkdir -p "${PILOT_ROOT}"
export FULL_INDEX_FILE PILOT_INDEX_FILE NORMAL_LABEL PILOT_VIDEOS PILOT_SEED
python - <<'PY'
import os
import random
from pathlib import Path

source = Path(os.environ["FULL_INDEX_FILE"])
target = Path(os.environ["PILOT_INDEX_FILE"])
normal_label = str(os.environ["NORMAL_LABEL"])
count = int(os.environ["PILOT_VIDEOS"])
seed = int(os.environ["PILOT_SEED"])
if not target.exists():
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    normal = [line for line in lines if line.split()[3] == normal_label]
    anomalous = [line for line in lines if line.split()[3] != normal_label]
    rng = random.Random(seed)
    rng.shuffle(normal)
    rng.shuffle(anomalous)
    normal_count = min(len(normal), count // 2)
    anomalous_count = min(len(anomalous), count - normal_count)
    chosen = normal[:normal_count] + anomalous[:anomalous_count]
    if len(chosen) < count:
        used = set(chosen)
        chosen.extend(line for line in lines if line not in used)
        chosen = chosen[:count]
    rng.shuffle(chosen)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{line}\n" for line in chosen))
print(f"Pilot index: {target} ({len(target.read_text().splitlines())} videos)")
PY

echo "Evaluating E0 on the identical pilot subset"
python -m src.eval \
  --root_path "${ROOT_PATH}" \
  --annotationfile_path "${PILOT_INDEX_FILE}" \
  --temporal_annotation_file "${TEMPORAL_ANNOTATION_FILE}" \
  --scores_dir "${E0_SCORES_DIR}" \
  --output_dir "${PILOT_ROOT}/e0_metrics" \
  --frame_interval 16 \
  --normal_label "${NORMAL_LABEL}"

for experiment in d1 shuffle noalign; do
  output_dir="${PILOT_ROOT}/scores_${experiment}"
  echo
  echo "Running pilot experiment: ${experiment}"
  GPU_IDS="${GPU_IDS}" \
  DATASET_DIR="${DATASET_DIR}" \
  VIDEO_DIR="${VIDEO_DIR}" \
  INDEX_FILE="${PILOT_INDEX_FILE}" \
  EVAL_ANNOTATION_FILE="${PILOT_INDEX_FILE}" \
  TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE}" \
  ROOT_PATH="${ROOT_PATH}" \
  OUTPUT_DIR="${output_dir}" \
  NORMAL_LABEL="${NORMAL_LABEL}" \
  PRECISE_TIME="${PRECISE_TIME}" \
  MOTION_FPS="${MOTION_FPS}" \
  SAVE_MOTION_DETAILS="${SAVE_MOTION_DETAILS}" \
  EXPERIMENT="${experiment}" \
  bash scripts/run_video_sdee_rebalanced.sh
done

echo
echo "E5-SDEE pilot comparison"
for item in \
  "E0:${PILOT_ROOT}/e0_metrics" \
  "E5-D1:${PILOT_ROOT}/scores_d1/metrics" \
  "Shuffle:${PILOT_ROOT}/scores_shuffle/metrics" \
  "NoAlign:${PILOT_ROOT}/scores_noalign/metrics"; do
  name="${item%%:*}"
  metric_dir="${item#*:}"
  printf '%-10s ROC=%s PR=%s F1=%s\n' \
    "${name}" \
    "$(tr -d '[:space:]' < "${metric_dir}/roc_auc.txt")" \
    "$(tr -d '[:space:]' < "${metric_dir}/pr_auc.txt")" \
    "$(tr -d '[:space:]' < "${metric_dir}/max_f1.txt")"
done
