#!/usr/bin/env bash
set -euo pipefail

# Caption-E0 ablation: reuse captions, replace visual tokens with caption text,
# and keep the ten cumulative YES/NO likelihood thresholds and PAVA integral.
# One Llama worker is assigned to each dataset.  UCF/MSAD share GPU 0;
# XD-Violence uses GPU 1 because it has substantially more clips.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
stty -tostop 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-.}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LLM="${LLM:-./libs/llama/llama3.1-8b}"
OUT_ROOT="${OUT_ROOT:-./data}"
COMMON=(
  --ckpt_dir "${LLM}"
  --tokenizer_path "${LLM}/tokenizer.model"
  --temperature "${TEMPERATURE:-1.0}"
  --max_seq_len "${MAX_SEQ_LEN:-1024}"
  --max_batch_size 2
  --resume
)

run_one() {
  local name="$1"
  local dataset="$2"
  local index_file="$3"
  local captions_dir="$4"
  local port="$5"
  local position="$6"
  local gpu="$7"
  local num_jobs="$8"
  local job_index="$9"
  local output_dir="${OUT_ROOT}/${dataset}/scores/caption_e0_likelihood_stride16"
  mkdir -p "${output_dir}"
  echo "Starting Caption-E0 ${name} on physical GPU ${gpu} -> ${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" torchrun --nproc_per_node=1 --nnodes=1 \
    --master_port="${port}" -m src.caption_cumulative_likelihood \
    --index_file "${index_file}" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}" \
    "${COMMON[@]}" \
    --num_jobs "${num_jobs}" \
    --job_index "${job_index}" \
    --progress_label "${name}" \
    --progress_position "${position}" \
    >/dev/null
}

pids=()
run_one UCF-Crime ucf_crime \
  ./data/ucf_crime/annotations/test.txt \
  ./data/ucf_crime/captions/video_llama3_json_results 29601 0 0 1 0 &
pids+=("$!")
run_one MSAD MSAD \
  ./data/MSAD/annotations/test.txt \
  ./data/MSAD/captions/video_llama3_json_results 29602 1 0 1 0 &
pids+=("$!")
run_one XD-Violence-1/2 xd_violence \
  ./data/xd_violence/annotations/test.txt \
  ./data/xd_violence/captions/video_llama3_json_results 29603 2 1 2 0 &
pids+=("$!")
run_one XD-Violence-2/2 xd_violence \
  ./data/xd_violence/annotations/test.txt \
  ./data/xd_violence/captions/video_llama3_json_results 29604 3 1 2 1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
