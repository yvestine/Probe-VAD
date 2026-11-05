#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/research_code"
export PYTHONPATH="$REPO/research_code${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-smoke}"
[[ "$MODE" == smoke || "$MODE" == full ]] || { echo 'Use smoke or full'; exit 2; }
: "${VIDEO_DIR:?Set absolute path to MSAD videos}"
: "${INDEX_FILE:?Set absolute path to full 360-video index}"
: "${TEMPORAL_ANNOTATION_FILE:?Set absolute path to full temporal annotations}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/results/experiments/MSAD/factorized_flat_e2_11class}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
# Same dataset and normal-label protocol as the requested matched experiment.
"$PYTHON_BIN" - "$VIDEO_DIR" "$INDEX_FILE" "$TEMPORAL_ANNOTATION_FILE" <<'PY'
import sys
from pathlib import Path
import torch
from src.video_score_utils import resolve_video_path, get_video_info, output_stem
rows=[[l.split() for l in Path(p).read_text().splitlines() if l.strip()] for p in sys.argv[2:]]
names=[{output_stem(r[0]) for r in rs} for rs in rows]
if not (len(rows[0])==len(rows[1])==len(names[0])==len(names[1])==360 and names[0]==names[1]):
    raise ValueError('Require matching complete 360-video index and temporal annotations')
if sum(r[3]=='0' for r in rows[0])!=120:
    raise ValueError('Expected 120 normal videos labeled 0; verify annotation protocol')
for row in rows[0]:
    info=get_video_info(resolve_video_path(Path(sys.argv[1]),row[0]))
    if int(row[1])!=0 or info.frame_count!=int(row[2])+1:
        raise ValueError(f'Full-video frame-count mismatch: {row[0]}')
if not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA required for real model smoke/full')
PY
COMMON=(--video_dir "$VIDEO_DIR" --index_file "$INDEX_FILE" --model_path "$MODEL_PATH"
        --device cuda:0 --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
        --threshold_batch_size "${THRESHOLD_BATCH_SIZE:-10}"
        --checkpoint_interval "${CHECKPOINT_INTERVAL:-20}")
if [[ "$MODE" == smoke ]]; then
  "$PYTHON_BIN" -m src.video_factorized_flat_e2_score "${COMMON[@]}" --output_dir "${OUTPUT_DIR}_smoke" --smoke
else
  "$PYTHON_BIN" -m src.video_factorized_flat_e2_score "${COMMON[@]}" --output_dir "${OUTPUT_DIR}_smoke" --smoke --validate_only
  "$PYTHON_BIN" -m src.video_factorized_flat_e2_score "${COMMON[@]}" --output_dir "$OUTPUT_DIR"
  # Coverage gate above must pass before the UNMODIFIED original evaluator runs.
  "$PYTHON_BIN" -m src.eval --root_path "${ROOT_PATH:-$VIDEO_DIR}" \
    --annotationfile_path "$INDEX_FILE" --temporal_annotation_file "$TEMPORAL_ANNOTATION_FILE" \
    --scores_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR/metrics_360" \
    --frame_interval 16 --normal_label 0 --smoothing_sigma 10
fi
