#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib_dataset_paths.sh"

# Actual five-threshold E0 ablation. One worker per dataset; existing E0
# directories are never modified.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
GPU_IDS="${GPU_IDS:-3,5,6}"
IFS=',' read -r GPU_UCF GPU_MSAD GPU_XD <<< "${GPU_IDS}"
THRESHOLDS="${THRESHOLDS:-0.2,0.4,0.6,0.8,1.0}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

run_one() {
  local dataset="$1" video_dir="$2" index_file="$3" output_dir="$4" gpu="$5"
  local precise_time="$6" num_jobs="$7" job_index="$8" position="$9"
  mkdir -p "${output_dir}"
  local precise_flag="--no-precise_time"
  [[ "${precise_time}" == "1" ]] && precise_flag="--precise_time"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m src.video_cumulative_threshold5 \
    --video_dir "${video_dir}" \
    --index_file "${index_file}" \
    --output_dir "${output_dir}" \
    --model_path "${MODEL_PATH}" \
    --device cuda:0 \
    --thresholds "${THRESHOLDS}" \
    --threshold_batch_size "${THRESHOLD_BATCH_SIZE}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    "${precise_flag}" \
    --num_jobs "${num_jobs}" --job_index "${job_index}" \
    --save_threshold_details --resume --log_level WARNING &
}

OUT_UCF="./data/ucf_crime/scores/videollama3_cumulative_likelihood_threshold5_actual_stride16"
OUT_MSAD="./data/MSAD/scores/videollama3_cumulative_likelihood_threshold5_actual_stride16"
OUT_XD="./data/xd_violence/scores/videollama3_cumulative_likelihood_threshold5_actual_stride16"

MSAD_INDEX="./data/MSAD/annotations/test.txt"
MSAD_TEMPORAL="./data/MSAD/annotations/msad_anomaly_index.txt"

pids=()
run_one ucf_crime ./data/ucf_crime/videos ./data/ucf_crime/annotations/test.txt \
  "${OUT_UCF}" "${GPU_UCF}" 0 2 0 0
pids+=("$!")
run_one ucf_crime ./data/ucf_crime/videos ./data/ucf_crime/annotations/test.txt \
  "${OUT_UCF}" "${GPU_UCF}" 0 2 1 1
pids+=("$!")
run_one MSAD ./data/MSAD/videos "${MSAD_INDEX}" \
  "${OUT_MSAD}" "${GPU_MSAD}" 1 1 0 2
pids+=("$!")
run_one xd_violence ./data/xd_violence/videos ./data/xd_violence/annotations/test.txt \
  "${OUT_XD}" "${GPU_XD}" 0 2 0 3
pids+=("$!")
run_one xd_violence ./data/xd_violence/videos ./data/xd_violence/annotations/test.txt \
  "${OUT_XD}" "${GPU_XD}" 0 2 1 4
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

eval_one() {
  local dataset="$1" root="$2" index="$3" temporal="$4" normal="$5" output_dir="$6"
  local metrics_dir="${output_dir}/metrics_official"
  [[ "${dataset}" == "MSAD" ]] && metrics_dir="${output_dir}/metrics_360"
  "${PYTHON_BIN}" -u -m src.eval \
    --root_path "${root}" \
    --annotationfile_path "${index}" \
    --temporal_annotation_file "${temporal}" \
    --scores_dir "${output_dir}" \
    --output_dir "${metrics_dir}" \
    --frame_interval 16 --normal_label "${normal}" --smoothing_sigma 10
}
eval_one ucf_crime ./data/ucf_crime/frames ./data/ucf_crime/annotations/test.txt \
  ./data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt 7 "${OUT_UCF}"
eval_one MSAD ./data/MSAD/frames "${MSAD_INDEX}" \
  "${MSAD_TEMPORAL}" 7 "${OUT_MSAD}"
eval_one xd_violence ./data/xd_violence/frames ./data/xd_violence/annotations/test.txt \
  ./data/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt 4 "${OUT_XD}"

echo "Actual five-threshold metrics"
for d in "${OUT_UCF}" "${OUT_MSAD}" "${OUT_XD}"; do
  echo "${d}"
  metric_dir="${d}/metrics_official"
  [[ "${d}" == "${OUT_MSAD}" ]] && metric_dir="${d}/metrics_360"
  printf 'ROC-AUC: %s\n' "$(tr -d '[:space:]' < "${metric_dir}/roc_auc.txt")"
  printf 'PR-AUC:  %s\n' "$(tr -d '[:space:]' < "${metric_dir}/pr_auc.txt")"
  printf 'Max-F1:  %s\n' "$(tr -d '[:space:]' < "${metric_dir}/max_f1.txt")"
done
