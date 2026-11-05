#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/research_code"
export PYTHONPATH="$REPO/research_code${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MODE="${1:-smoke}"
[[ "$MODE" == check || "$MODE" == smoke || "$MODE" == full ]] || { echo 'Use check, smoke or full'; exit 2; }
"$PYTHON_BIN" -m src.factorized_cuda_preflight --attn_implementation "$ATTN_IMPLEMENTATION"
[[ "$MODE" != check ]] || exit 0
: "${VIDEO_DIR:?Set VIDEO_DIR to full MSAD videos}"
: "${INDEX_FILE:?Set INDEX_FILE to full 360-video index}"
: "${TEMPORAL_ANNOTATION_FILE:?Set TEMPORAL_ANNOTATION_FILE to full 360-video annotations}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results/experiments/MSAD}"

# Do not silently substitute the public 240-video protocol.
"$PYTHON_BIN" - "$INDEX_FILE" "$TEMPORAL_ANNOTATION_FILE" "$VIDEO_DIR" <<'PY'
import sys
from pathlib import Path
rows=[[l.split() for l in Path(p).read_text().splitlines() if l.strip()] for p in sys.argv[1:3]]
names=[{Path(r[0]).name.removesuffix('.mp4') for r in rs} for rs in rows]
assert all(len(rs)==360 for rs in rows) and len(names[0])==360 and names[0]==names[1], 'Need matching 360-video annotations'
assert sum(r[3]=='0' for r in rows[0])==120, 'Require 120 normal videos labeled 0'
from src.video_score_utils import resolve_video_path, get_video_info
for row in rows[0]:
    assert int(row[1])==0, 'Full-video index required'
    info=get_video_info(resolve_video_path(Path(sys.argv[3]),row[0]))
    assert info.frame_count==int(row[2])+1, f'Frame count mismatch: {row[0]}'
import torch
assert torch.cuda.is_available(), 'Real model smoke/full requires CUDA'
PY
COMMON=(--video_dir "$VIDEO_DIR" --index_file "$INDEX_FILE" --model_path "$MODEL_PATH")
PROMPT="$("$PYTHON_BIN" -c 'from src.video_cumulative_score import CUMULATIVE_PROMPT; print(CUMULATIVE_PROMPT)')"
EXTRA=()
if [[ "$MODE" == smoke ]]; then
  RESULTS_ROOT="$RESULTS_ROOT/_smoke_factorized_flat_11class"
  EXTRA=(--smoke)
else
  "$PYTHON_BIN" - "$RESULTS_ROOT/_smoke_factorized_flat_11class" "$MODEL_PATH" "$ATTN_IMPLEMENTATION" <<'PY'
import hashlib,json,sys
from pathlib import Path
from src import video_factorized_flat_score as f
from src import video_score_utils as u
source=hashlib.sha256(Path(f.__file__).read_bytes()+Path(u.__file__).read_bytes()+Path(f.__file__).with_name('factorized_sequence_likelihood.py').read_bytes()).hexdigest()
for method in f.METHODS:
    m=json.loads((Path(sys.argv[1])/method/'_experiment.json').read_text())
    assert m['model']==sys.argv[2] and m['source_sha256']==source and m['attn_implementation']==sys.argv[3], 'Repeat smoke with current model/code'
PY
  # Revalidate smoke scores and frame identities; a stale marker alone is insufficient.
  "$PYTHON_BIN" -m src.eval_factorized_flat --video_dir "$VIDEO_DIR" --index_file "$INDEX_FILE" \
    --temporal_annotation_file "$TEMPORAL_ANNOTATION_FILE" \
    --results_root "$RESULTS_ROOT/_smoke_factorized_flat_11class" --smoke
fi
for method in factorized_flat_11class flat_11class_matched probe_vad_matched; do
  "$PYTHON_BIN" -m src.video_factorized_flat_score "${COMMON[@]}" --method "$method" \
    --attn_implementation "$ATTN_IMPLEMENTATION" --device cuda:0 \
    --prompt "$PROMPT" --output_dir "$RESULTS_ROOT/$method" \
    --threshold_batch_size "${THRESHOLD_BATCH_SIZE:-11}" "${EXTRA[@]}"
done
"$PYTHON_BIN" -m src.eval_factorized_flat --video_dir "$VIDEO_DIR" --index_file "$INDEX_FILE" \
  --temporal_annotation_file "$TEMPORAL_ANNOTATION_FILE" --results_root "$RESULTS_ROOT" \
  --bootstrap "${BOOTSTRAP_REPEATS:-1000}" "${EXTRA[@]}"
