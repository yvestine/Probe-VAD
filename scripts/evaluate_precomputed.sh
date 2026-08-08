#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

selection="${1:-all}"

evaluate_one() {
  local dataset="$1"
  local normal_label="$2"
  local frame_root="$3"
  echo "Evaluating bundled ${dataset} scores"
  python -m covas_vad.evaluation \
    --root_path "${frame_root}" \
    --annotationfile_path "./results/${dataset}/annotations/test.txt" \
    --temporal_annotation_file "./results/${dataset}/annotations/temporal_annotations.txt" \
    --scores_dir "./results/${dataset}/scores" \
    --output_dir "./results/${dataset}/metrics_reproduced" \
    --frame_interval 16 \
    --normal_label "${normal_label}"
}

case "${selection}" in
  ucf_crime)
    evaluate_one ucf_crime 7 ./data/ucf_crime/frames
    ;;
  msad)
    evaluate_one msad 0 ./data/MSAD/frames
    ;;
  xd_violence)
    evaluate_one xd_violence 4 ./data/xd_violence/frames
    ;;
  all)
    evaluate_one ucf_crime 7 ./data/ucf_crime/frames
    evaluate_one msad 0 ./data/MSAD/frames
    evaluate_one xd_violence 4 ./data/xd_violence/frames
    ;;
  *)
    echo "Usage: bash scripts/evaluate_precomputed.sh [all|ucf_crime|msad|xd_violence]" >&2
    exit 2
    ;;
esac

