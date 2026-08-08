#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASET_DIR="${DATASET_DIR:-./data/ucf_crime}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
EXPERIMENT="${EXPERIMENT:-d1}"
case "${EXPERIMENT}" in
  d1)
    MOTION_MODE="aligned"
    SHUFFLE_MOTION=0
    DEFAULT_NAME="videollama3_e5_sdee_d1"
    ;;
  shuffle)
    MOTION_MODE="aligned"
    SHUFFLE_MOTION=1
    DEFAULT_NAME="videollama3_e5_sdee_d1_shuffle"
    ;;
  noalign)
    MOTION_MODE="noalign"
    SHUFFLE_MOTION=0
    DEFAULT_NAME="videollama3_e5_sdee_d1_noalign"
    ;;
  *)
    echo "EXPERIMENT must be d1, shuffle, or noalign" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/scores/${DEFAULT_NAME}}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
GPU_IDS="${GPU_IDS:-0}"
ROOT_PATH="${ROOT_PATH:-${DATASET_DIR}/frames}"
EVAL_ANNOTATION_FILE="${EVAL_ANNOTATION_FILE:-${INDEX_FILE}}"
TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE:-${DATASET_DIR}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt}"
NORMAL_LABEL="${NORMAL_LABEL:-7}"
FRAME_INTERVAL="${FRAME_INTERVAL:-16}"
WINDOW_SECONDS="${WINDOW_SECONDS:-10}"
TEMPORAL_BINS="${TEMPORAL_BINS:-5}"
MOTION_FPS="${MOTION_FPS:-8}"
DECODE_MAX_FRAMES="${DECODE_MAX_FRAMES:-96}"
PRECISE_TIME="${PRECISE_TIME:-0}"
LIKELIHOOD_TEMPERATURE="${LIKELIHOOD_TEMPERATURE:-1.0}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-10}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"
SAVE_THRESHOLD_DETAILS="${SAVE_THRESHOLD_DETAILS:-0}"
SAVE_MOTION_DETAILS="${SAVE_MOTION_DETAILS:-0}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_DIR}/_remaining_shards}"

case "$(basename "${OUTPUT_DIR%/}")" in
  videollama3_cumulative_likelihood|videollama3_cumulative_likelihood_optimized)
    echo "Refusing to overwrite protected E0 directory: ${OUTPUT_DIR}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_JOBS="${#GPUS[@]}"
mkdir -p "${OUTPUT_DIR}" "${SHARD_DIR}"
export VIDEO_DIR INDEX_FILE OUTPUT_DIR SHARD_DIR NUM_JOBS
export FRAME_INTERVAL WINDOW_SECONDS SAVE_THRESHOLD_DETAILS SAVE_MOTION_DETAILS

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
jobs = int(os.environ["NUM_JOBS"])
frame_interval = int(os.environ["FRAME_INTERVAL"])
window_seconds = float(os.environ["WINDOW_SECONDS"])


def load(path):
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
    scores = load(output_dir / f"{stem}.json")
    errors = load(output_dir / "_errors" / f"{stem}.json")
    completed = expected.intersection(scores).difference(errors)
    if os.environ["SAVE_THRESHOLD_DETAILS"] != "0":
        details = load(output_dir / "_threshold_details" / f"{stem}.json")
        completed.intersection_update(details)
    if os.environ["SAVE_MOTION_DETAILS"] != "0":
        details = load(output_dir / "_motion_details" / f"{stem}.json")
        completed.intersection_update(details)
    work = len(expected) - len(completed)
    if work:
        remaining.append((work, name))

bins = [[] for _ in range(jobs)]
loads = [0] * jobs
for work, name in sorted(remaining, reverse=True):
    job = min(range(jobs), key=loads.__getitem__)
    bins[job].append(name)
    loads[job] += work
for job, names in enumerate(bins):
    path = shard_dir / f"job_{job}.txt"
    path.write_text("".join(f"{name}\n" for name in names))
    print(
        f"job {job}: {len(names)} videos, {loads[job]} remaining windows, "
        f"index={path}"
    )
print(f"total remaining windows: {sum(loads)}")
PY

precise_flag="--precise_time"
[[ "${PRECISE_TIME}" == "0" ]] && precise_flag="--no-precise_time"
prefix_flag="--prefix_cache"
[[ "${PREFIX_CACHE}" == "0" ]] && prefix_flag="--no-prefix_cache"
shuffle_flag="--shuffle_motion"
[[ "${SHUFFLE_MOTION}" == "0" ]] && shuffle_flag="--no-shuffle_motion"
threshold_flag="--save_threshold_details"
[[ "${SAVE_THRESHOLD_DETAILS}" == "0" ]] && threshold_flag="--no-save_threshold_details"
motion_flag="--save_motion_details"
[[ "${SAVE_MOTION_DETAILS}" == "0" ]] && motion_flag="--no-save_motion_details"

echo "Starting E5-SDEE experiment=${EXPERIMENT} on GPUs: ${GPU_IDS}"
pids=()
for job in "${!GPUS[@]}"; do
  shard="${SHARD_DIR}/job_${job}.txt"
  if [[ ! -s "${shard}" ]]; then
    echo "Job ${job}: no remaining work"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[$job]}" python -u -m src.video_sdee_score \
    --video_dir "${VIDEO_DIR}" \
    --index_file "${shard}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --device cuda:0 \
    --frame_interval "${FRAME_INTERVAL}" \
    --window_seconds "${WINDOW_SECONDS}" \
    --temporal_bins "${TEMPORAL_BINS}" \
    --motion_fps "${MOTION_FPS}" \
    --decode_max_frames "${DECODE_MAX_FRAMES}" \
    --motion_mode "${MOTION_MODE}" \
    "${shuffle_flag}" \
    "${precise_flag}" \
    --likelihood_temperature "${LIKELIHOOD_TEMPERATURE}" \
    --threshold_batch_size "${THRESHOLD_BATCH_SIZE}" \
    "${prefix_flag}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    "${threshold_flag}" \
    "${motion_flag}" \
    --resume &
  pids+=("$!")
done
if ((${#pids[@]})); then
  wait "${pids[@]}"
fi

python -m src.eval \
  --root_path "${ROOT_PATH}" \
  --annotationfile_path "${EVAL_ANNOTATION_FILE}" \
  --temporal_annotation_file "${TEMPORAL_ANNOTATION_FILE}" \
  --scores_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/metrics" \
  --frame_interval "${FRAME_INTERVAL}" \
  --normal_label "${NORMAL_LABEL}"

echo
echo "E5-SDEE ${EXPERIMENT} metrics"
printf 'ROC-AUC: %s\n' "$(tr -d '[:space:]' < "${OUTPUT_DIR}/metrics/roc_auc.txt")"
printf 'PR-AUC:  %s\n' "$(tr -d '[:space:]' < "${OUTPUT_DIR}/metrics/pr_auc.txt")"
printf 'Max-F1:  %s\n' "$(tr -d '[:space:]' < "${OUTPUT_DIR}/metrics/max_f1.txt")"
