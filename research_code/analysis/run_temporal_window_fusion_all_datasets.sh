#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FUSION_MODES="${FUSION_MODES:-overlap_mean,overlap_logit,adaptive}"
IFS=',' read -r -a MODES <<< "${FUSION_MODES}"

run_dataset() {
  local dataset_dir="$1"
  local scores_dir="$2"
  local index_file="$3"
  local temporal_file="$4"
  local normal_label="$5"
  local dataset_name="$6"

  for mode in "${MODES[@]}"; do
    local output_dir="${dataset_dir}/scores/videollama3_e0_temporal_fusion_${mode}"
    echo
    echo "[$dataset_name] fusion_mode=$mode"
    "${PYTHON_BIN}" -u -m src.temporal_window_fusion \
      --scores_dir "${scores_dir}" \
      --video_dir "${dataset_dir}/videos" \
      --index_file "${index_file}" \
      --output_dir "${output_dir}" \
      --fusion_mode "${mode}" \
      --window_seconds 10 \
      --no-save_details \
      --log_level WARNING \
      --resume

    "${PYTHON_BIN}" -m src.eval \
      --root_path "${dataset_dir}/frames" \
      --annotationfile_path "${index_file}" \
      --temporal_annotation_file "${temporal_file}" \
      --scores_dir "${output_dir}" \
      --output_dir "${output_dir}/metrics_raw" \
      --frame_interval 16 \
      --normal_label "${normal_label}" \
      --no_smoothing

    "${PYTHON_BIN}" -m src.eval \
      --root_path "${dataset_dir}/frames" \
      --annotationfile_path "${index_file}" \
      --temporal_annotation_file "${temporal_file}" \
      --scores_dir "${output_dir}" \
      --output_dir "${output_dir}/metrics_sigma10" \
      --frame_interval 16 \
      --normal_label "${normal_label}" \
      --smoothing_sigma 10

    printf '[%s/%s raw] ROC=%s PR=%s F1=%s\n' \
      "${dataset_name}" "${mode}" \
      "$(tr -d '[:space:]' < "${output_dir}/metrics_raw/roc_auc.txt")" \
      "$(tr -d '[:space:]' < "${output_dir}/metrics_raw/pr_auc.txt")" \
      "$(tr -d '[:space:]' < "${output_dir}/metrics_raw/max_f1.txt")"
    printf '[%s/%s sigma10] ROC=%s PR=%s F1=%s\n' \
      "${dataset_name}" "${mode}" \
      "$(tr -d '[:space:]' < "${output_dir}/metrics_sigma10/roc_auc.txt")" \
      "$(tr -d '[:space:]' < "${output_dir}/metrics_sigma10/pr_auc.txt")" \
      "$(tr -d '[:space:]' < "${output_dir}/metrics_sigma10/max_f1.txt")"
  done
}

run_dataset \
  ./data/MSAD \
  ./data/MSAD/scores/videollama3_cumulative_likelihood_optimized \
  ./VADTree/dataset_info/MSAD/annotations/anomaly_test.txt \
  ./VADTree/dataset_info/MSAD/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
  0 \
  MSAD

run_dataset \
  ./data/ucf_crime \
  ./data/ucf_crime/scores/videollama3_cumulative_likelihood \
  ./VADTree/dataset_info/ucf_crime/annotations/anomaly_test.txt \
  ./VADTree/dataset_info/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
  7 \
  UCF-Crime

run_dataset \
  ./data/xd_violence \
  ./data/xd_violence/scores/videollama3_cumulative_likelihood_optimized \
  ./VADTree/dataset_info/xd_violence/annotations/anomaly_test.txt \
  ./VADTree/dataset_info/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt \
  4 \
  XD-Violence
