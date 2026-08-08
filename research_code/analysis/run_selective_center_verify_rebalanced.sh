#!/usr/bin/env bash
set -euo pipefail

# Selective center-4s verification. This script never overwrites or fuses E0.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASET_DIR="${DATASET_DIR:-./data/ucf_crime}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
INITIAL_SCORES_DIR="${INITIAL_SCORES_DIR:-${DATASET_DIR}/scores/videollama3_cumulative_likelihood}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/scores/videollama3_e0_selective_center4_verify_q85_q95}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${DATASET_DIR}/analysis/e0_selective_center4_verify_q85_q95}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
GPU_IDS="${GPU_IDS:-0}"
ROOT_PATH="${ROOT_PATH:-${DATASET_DIR}/frames}"
EVAL_ANNOTATION_FILE="${EVAL_ANNOTATION_FILE:-${INDEX_FILE}}"
TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE:-${DATASET_DIR}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt}"
NORMAL_LABEL="${NORMAL_LABEL:-7}"
LOWER_QUANTILE="${LOWER_QUANTILE:-0.85}"
UPPER_QUANTILE="${UPPER_QUANTILE:-0.95}"
VERIFY_SECONDS="${VERIFY_SECONDS:-4}"
SAMPLE_FPS="${SAMPLE_FPS:-2.5}"
MAX_FRAMES="${MAX_FRAMES:-10}"
PRECISE_TIME="${PRECISE_TIME:-0}"
LIKELIHOOD_TEMPERATURE="${LIKELIHOOD_TEMPERATURE:-1.0}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-10}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"
SAVE_THRESHOLD_DETAILS="${SAVE_THRESHOLD_DETAILS:-0}"
FRAME_INTERVAL="${FRAME_INTERVAL:-16}"
SMOOTHING_SIGMA="${SMOOTHING_SIGMA:-10}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_DIR}/_remaining_shards}"

if [[ "$(realpath -m "${OUTPUT_DIR}")" == "$(realpath -m "${INITIAL_SCORES_DIR}")" ]]; then
  echo "OUTPUT_DIR must differ from INITIAL_SCORES_DIR; refusing to overwrite E0." >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_JOBS="${#GPUS[@]}"
mkdir -p "${OUTPUT_DIR}" "${SHARD_DIR}" "${ANALYSIS_DIR}"

export VIDEO_DIR INDEX_FILE INITIAL_SCORES_DIR OUTPUT_DIR SHARD_DIR
export LOWER_QUANTILE UPPER_QUANTILE NUM_JOBS SAVE_THRESHOLD_DETAILS

python - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.video_score_utils import output_stem
from src.video_selective_center_verify import (
    ordered_initial_scores,
    select_percentile_band,
)

index_file = Path(os.environ["INDEX_FILE"])
initial_dir = Path(os.environ["INITIAL_SCORES_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
shard_dir = Path(os.environ["SHARD_DIR"])
lower = float(os.environ["LOWER_QUANTILE"])
upper = float(os.environ["UPPER_QUANTILE"])
num_jobs = int(os.environ["NUM_JOBS"])
need_threshold_details = os.environ["SAVE_THRESHOLD_DETAILS"] != "0"


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
    initial = load_dict(initial_dir / f"{stem}.json")
    if not initial:
        raise FileNotFoundError(f"missing initial E0 score JSON: {stem}")
    selected = select_percentile_band(
        ordered_initial_scores(initial),
        lower,
        upper,
    )
    expected = {str(item["center_frame"]) for item in selected}
    scores = load_dict(output_dir / f"{stem}.json")
    details = load_dict(output_dir / "_verification_details" / f"{stem}.json")
    errors = load_dict(output_dir / "_errors" / f"{stem}.json")
    completed = expected.intersection(scores).intersection(details).difference(errors)
    if need_threshold_details:
        threshold_details = load_dict(
            output_dir / "_threshold_details" / f"{stem}.json"
        )
        completed.intersection_update(threshold_details)
    work = len(expected) - len(completed)
    if work:
        remaining.append((work, name))

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
        f"{loads[job]} remaining verification clips, index={path}"
    )
print(
    f"total: {len(remaining)} unfinished videos, "
    f"{sum(loads)} remaining verification clips"
)
PY

prefix_cache_flag="--prefix_cache"
if [[ "${PREFIX_CACHE}" == "0" ]]; then
  prefix_cache_flag="--no-prefix_cache"
fi
precise_time_flag="--precise_time"
if [[ "${PRECISE_TIME}" == "0" ]]; then
  precise_time_flag="--no-precise_time"
fi
threshold_details_flag="--save_threshold_details"
if [[ "${SAVE_THRESHOLD_DETAILS}" == "0" ]]; then
  threshold_details_flag="--no-save_threshold_details"
fi

echo "Starting selective E0 center verification on GPUs: ${GPU_IDS}"
echo "Band: [${LOWER_QUANTILE}, ${UPPER_QUANTILE}); verify=${VERIFY_SECONDS}s; frames=${MAX_FRAMES}"

pids=()
for job in "${!GPUS[@]}"; do
  shard="${SHARD_DIR}/job_${job}.txt"
  if [[ ! -s "${shard}" ]]; then
    echo "Job ${job}: no remaining work"
    continue
  fi
  echo "Job ${job}: physical GPU ${GPUS[$job]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$job]}" python -u -m src.video_selective_center_verify \
    --video_dir "${VIDEO_DIR}" \
    --index_file "${shard}" \
    --initial_scores_dir "${INITIAL_SCORES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --device cuda:0 \
    --lower_quantile "${LOWER_QUANTILE}" \
    --upper_quantile "${UPPER_QUANTILE}" \
    --verify_seconds "${VERIFY_SECONDS}" \
    --sample_fps "${SAMPLE_FPS}" \
    --max_frames "${MAX_FRAMES}" \
    "${precise_time_flag}" \
    --likelihood_temperature "${LIKELIHOOD_TEMPERATURE}" \
    --threshold_batch_size "${THRESHOLD_BATCH_SIZE}" \
    "${prefix_cache_flag}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    "${threshold_details_flag}" \
    --resume &
  pids+=("$!")
done

if ((${#pids[@]})); then
  wait "${pids[@]}"
fi

echo
echo "Analyzing verification scores without fusion"
python -u -m src.analyze_selective_center_verify \
  --root_path "${ROOT_PATH}" \
  --annotation_file "${EVAL_ANNOTATION_FILE}" \
  --temporal_annotation_file "${TEMPORAL_ANNOTATION_FILE}" \
  --initial_scores_dir "${INITIAL_SCORES_DIR}" \
  --verify_scores_dir "${OUTPUT_DIR}" \
  --output_dir "${ANALYSIS_DIR}" \
  --normal_label "${NORMAL_LABEL}" \
  --frame_interval "${FRAME_INTERVAL}" \
  --smoothing_sigma "${SMOOTHING_SIGMA}"

echo
echo "Selective verification completed; no E0 scores were fused or overwritten."
echo "Verification scores: ${OUTPUT_DIR}"
echo "Diagnostic analysis: ${ANALYSIS_DIR}"
