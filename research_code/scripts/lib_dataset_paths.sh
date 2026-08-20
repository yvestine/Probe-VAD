#!/usr/bin/env bash

# Shared dataset-path defaults for experiment runners.
# MSAD has a complete 360-video protocol in data/MSAD. The bundled annotation
# files are selected implicitly for a full MSAD run.

covas_resolve_msad_paths() {
  local dataset_dir="${1:-${DATASET_DIR:-}}"
  if [[ "${dataset_dir%/}" == "./data/MSAD" || "${dataset_dir%/}" == "data/MSAD" || "${dataset_dir%/}" == "/workspace/gujiawei/URF-HVAA/data/MSAD" ]]; then
    if [[ -z "${INDEX_FILE:-}" || "${INDEX_FILE}" == *"results/msad/annotations"* ]]; then
      INDEX_FILE="${dataset_dir}/annotations/test.txt"
    fi
    if [[ -z "${EVAL_ANNOTATION_FILE:-}" || "${EVAL_ANNOTATION_FILE}" == *"results/msad/annotations"* ]]; then
      EVAL_ANNOTATION_FILE="${INDEX_FILE}"
    fi
    if [[ -z "${TEMPORAL_ANNOTATION_FILE:-}" || "${TEMPORAL_ANNOTATION_FILE}" == *"results/msad/annotations"* || "${TEMPORAL_ANNOTATION_FILE}" == *"Temporal_Anomaly_Annotation_for_Testing_Videos.txt"* ]]; then
      if [[ -f "${dataset_dir}/annotations/msad_anomaly_index.txt" ]]; then
        TEMPORAL_ANNOTATION_FILE="${dataset_dir}/annotations/msad_anomaly_index.txt"
      fi
    fi
  fi
  export INDEX_FILE EVAL_ANNOTATION_FILE TEMPORAL_ANNOTATION_FILE
}

covas_default_metrics_dir() {
  local output_dir="$1" dataset_dir="${2:-${DATASET_DIR:-}}" index_file="${3:-${INDEX_FILE:-}}"
  if [[ "${dataset_dir%/}" == *"MSAD" && -f "${index_file}" ]] && [[ "$(wc -l < "${index_file}")" -ge 360 ]]; then
    printf '%s/metrics_360\n' "${output_dir%/}"
  else
    printf '%s/metrics\n' "${output_dir%/}"
  fi
}
