#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
stty -tostop 2>/dev/null || true

GPU_IDS="${GPU_IDS:-0,0,1,1}"
DATASET_DIR="${DATASET_DIR:-./data/ucf_crime}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
CAPTIONS_DIR="${CAPTIONS_DIR:-${DATASET_DIR}/captions/video_llama3_json_results}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/scores/caption_e0_likelihood_stride16_rerun}"
LLM="${LLM:-./libs/llama/llama3.1-8b}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_JOBS="${#GPUS[@]}"
mkdir -p "${OUTPUT_DIR}"

pids=()
for job in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$job]}" torchrun --nproc_per_node=1 --nnodes=1 \
    --master_port=$((29711 + job)) -m src.caption_cumulative_likelihood \
    --index_file "${INDEX_FILE}" \
    --captions_dir "${CAPTIONS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --ckpt_dir "${LLM}" \
    --tokenizer_path "${LLM}/tokenizer.model" \
    --temperature 1.0 \
    --max_seq_len 1024 \
    --max_batch_size 2 \
    --num_jobs "${NUM_JOBS}" \
    --job_index "${job}" \
    --progress_label "Caption-UCF-${job}" \
    --progress_position "${job}" \
    --resume &
  pids+=("$!")
done

wait "${pids[@]}"

python -m src.eval_interpolated \
  --root_path "${DATASET_DIR}/frames" \
  --annotationfile_path "${INDEX_FILE}" \
  --temporal_annotation_file "${DATASET_DIR}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt" \
  --scores_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/metrics" \
  --scoring_interval 16 \
  --output_interval 16 \
  --normal_label 7
