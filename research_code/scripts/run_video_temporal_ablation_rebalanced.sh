#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
stty -tostop 2>/dev/null || true

MODE="${MODE:?set MODE=shuffle or MODE=single}"
DATASET_DIR="${DATASET_DIR:-./data/ucf_crime}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
GPU_IDS="${GPU_IDS:-4,4,5,5}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/scores/videollama3_e0_${MODE}_stride16}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
SHUFFLE_SEED="${SHUFFLE_SEED:-17}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_JOBS="${#GPUS[@]}"
mkdir -p "${OUTPUT_DIR}"
pids=()
for job in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$job]}" python -u -m src.video_temporal_order_ablation \
    --video_dir "${VIDEO_DIR}" \
    --index_file "${INDEX_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --temporal_mode "${MODE}" \
    --shuffle_seed "${SHUFFLE_SEED}" \
    --model_path "${MODEL_PATH}" \
    --frame_interval 16 \
    --window_seconds 10 \
    --sample_fps 2 \
    --max_frames 10 \
    --threshold_batch_size 10 \
    --prefix_cache \
    --monotonic_projection \
    --checkpoint_interval 20 \
    --resume \
    --num_jobs "${NUM_JOBS}" \
    --job_index "${job}" &
  pids+=("$!")
done
wait "${pids[@]}"

python -m src.eval_interpolated \
  --root_path "${DATASET_DIR}/frames" \
  --annotationfile_path "${INDEX_FILE}" \
  --temporal_annotation_file "${DATASET_DIR}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt" \
  --scores_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/metrics" \
  --scoring_interval 16 --output_interval 16 --normal_label 7
