#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASET_DIR="${DATASET_DIR:-./data/ucf_crime}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/scores/videollama3_cumulative_likelihood}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
GPU_IDS="${GPU_IDS:-0,2,3,6}"
ROOT_PATH="${ROOT_PATH:-${DATASET_DIR}/frames}"
EVAL_ANNOTATION_FILE="${EVAL_ANNOTATION_FILE:-${INDEX_FILE}}"
TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE:-${DATASET_DIR}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt}"
FRAME_INTERVAL="${FRAME_INTERVAL:-16}"
WINDOW_SECONDS="${WINDOW_SECONDS:-10}"
SAMPLE_FPS="${SAMPLE_FPS:-2}"
MAX_FRAMES="${MAX_FRAMES:-10}"
PRECISE_TIME="${PRECISE_TIME:-0}"
LIKELIHOOD_TEMPERATURE="${LIKELIHOOD_TEMPERATURE:-1.0}"
NORMAL_LABEL="${NORMAL_LABEL:-7}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-10}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"
SAVE_THRESHOLD_DETAILS="${SAVE_THRESHOLD_DETAILS:-0}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_DIR}/_remaining_shards}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_JOBS="${#GPUS[@]}"
mkdir -p "${OUTPUT_DIR}" "${SHARD_DIR}"

export DATASET_DIR VIDEO_DIR INDEX_FILE OUTPUT_DIR FRAME_INTERVAL WINDOW_SECONDS
export SHARD_DIR NUM_JOBS SAVE_THRESHOLD_DETAILS

python - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.video_score_utils import (
    get_video_info,
    iter_video_windows,
    output_stem,
    resolve_video_path,
)

video_dir = Path(os.environ["VIDEO_DIR"])
index_file = Path(os.environ["INDEX_FILE"])
output_dir = Path(os.environ["OUTPUT_DIR"])
shard_dir = Path(os.environ["SHARD_DIR"])
frame_interval = int(os.environ["FRAME_INTERVAL"])
window_seconds = float(os.environ["WINDOW_SECONDS"])
num_jobs = int(os.environ["NUM_JOBS"])


def load_dict(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


remaining = []
for line in index_file.read_text().splitlines():
    if not line.strip():
        continue
    name = line.split()[0]
    stem = output_stem(name)
    info = get_video_info(resolve_video_path(video_dir, name))
    expected = {
        str(window.center_frame)
        for window in iter_video_windows(info, frame_interval, window_seconds)
    }
    scores = load_dict(output_dir / f"{stem}.json")
    errors = load_dict(output_dir / "_errors" / f"{stem}.json")
    completed = expected.intersection(scores).difference(errors)
    if os.environ["SAVE_THRESHOLD_DETAILS"] != "0":
        details = load_dict(output_dir / "_threshold_details" / f"{stem}.json")
        completed.intersection_update(details)
    work = len(expected) - len(completed)
    if work:
        remaining.append((work, name))

# Longest-processing-time allocation balances the estimated remaining clip count.
bins = [[] for _ in range(num_jobs)]
loads = [0] * num_jobs
for work, name in sorted(remaining, reverse=True):
    job = min(range(num_jobs), key=loads.__getitem__)
    bins[job].append(name)
    loads[job] += work

for job, names in enumerate(bins):
    path = shard_dir / f"job_{job}.txt"
    path.write_text("".join(f"{name}\n" for name in names))
    print(
        f"job {job}: {len(names)} unfinished videos, "
        f"{loads[job]} remaining clips, index={path}"
    )
print(f"total: {len(remaining)} unfinished videos, {sum(loads)} remaining clips")
PY

echo "Starting balanced cumulative scoring on GPUs: ${GPU_IDS}"
prefix_cache_flag="--prefix_cache"
if [[ "${PREFIX_CACHE}" == "0" ]]; then
  prefix_cache_flag="--no-prefix_cache"
fi
details_flag="--save_threshold_details"
if [[ "${SAVE_THRESHOLD_DETAILS}" == "0" ]]; then
  details_flag="--no-save_threshold_details"
fi
precise_time_flag="--precise_time"
if [[ "${PRECISE_TIME}" == "0" ]]; then
  precise_time_flag="--no-precise_time"
fi
pids=()
for job in "${!GPUS[@]}"; do
  shard="${SHARD_DIR}/job_${job}.txt"
  if [[ ! -s "${shard}" ]]; then
    echo "Job ${job}: no remaining work"
    continue
  fi
  echo "Job ${job}: physical GPU ${GPUS[$job]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$job]}" python -u -m src.video_cumulative_score \
    --video_dir "${VIDEO_DIR}" \
    --index_file "${shard}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --device cuda:0 \
    --frame_interval "${FRAME_INTERVAL}" \
    --window_seconds "${WINDOW_SECONDS}" \
    --sample_fps "${SAMPLE_FPS}" \
    --max_frames "${MAX_FRAMES}" \
    "${precise_time_flag}" \
    --likelihood_temperature "${LIKELIHOOD_TEMPERATURE}" \
    --optimized \
    --threshold_batch_size "${THRESHOLD_BATCH_SIZE}" \
    "${prefix_cache_flag}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    "${details_flag}" \
    --monotonic_projection \
    --resume &
  pids+=("$!")
done

if ((${#pids[@]})); then
  wait "${pids[@]}"
fi

echo "Evaluating cumulative likelihood scores"
python -m src.eval \
  --root_path "${ROOT_PATH}" \
  --annotationfile_path "${EVAL_ANNOTATION_FILE}" \
  --temporal_annotation_file "${TEMPORAL_ANNOTATION_FILE}" \
  --scores_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/metrics" \
  --frame_interval "${FRAME_INTERVAL}" \
  --normal_label "${NORMAL_LABEL}"

echo
echo "Cumulative likelihood metrics"
printf 'ROC-AUC: %s\n' "$(tr -d '[:space:]' < "${OUTPUT_DIR}/metrics/roc_auc.txt")"
printf 'PR-AUC:  %s\n' "$(tr -d '[:space:]' < "${OUTPUT_DIR}/metrics/pr_auc.txt")"
printf 'Max-F1:  %s\n' "$(tr -d '[:space:]' < "${OUTPUT_DIR}/metrics/max_f1.txt")"
echo "Metrics saved under: ${OUTPUT_DIR}/metrics"
