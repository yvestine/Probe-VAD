#!/usr/bin/env bash
set -euo pipefail

# Complete the MSAD experiments on the 360-video protocol. Every stage is
# resumable: existing score JSONs are retained and only missing clips are
# processed. Stages are intentionally sequential so one model copy per GPU is
# active at a time and the output directories cannot interfere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"
stty -tostop 2>/dev/null || true

DATASET_DIR="${DATASET_DIR:-./data/MSAD}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_DIR}/videos}"
INDEX_FILE="${INDEX_FILE:-${DATASET_DIR}/annotations/test.txt}"
TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE:-${DATASET_DIR}/annotations/msad_anomaly_index.txt}"
ROOT_PATH="${ROOT_PATH:-${DATASET_DIR}/frames}"
MODEL_PATH="${MODEL_PATH:-DAMO-NLP-SG/VideoLLaMA3-7B}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/gujiawei/miniconda3/envs/VAA/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "VAA Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
"${PYTHON_BIN}" -c 'import accelerate; print(f"accelerate={accelerate.__version__}")' || {
  echo "Selected Python lacks accelerate>=0.26: ${PYTHON_BIN}" >&2
  exit 2
}
# Child wrappers historically invoke `python`; put the selected VAA runtime
# first so every worker uses the same environment as the CUDA preflight.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHON_BIN
# MSAD is high-resolution; one worker is the safe default. Override with a
# comma-separated list only after checking free VRAM (e.g. GPU_IDS=2,3).
GPU_IDS="${GPU_IDS:-0,1,2}"
NORMAL_LABEL="${NORMAL_LABEL:-7}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-10}"
MAX_FRAMES="${MAX_FRAMES:-10}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
# max_frames is an experimental invariant and is never changed by OOM retry.
export THRESHOLD_BATCH_SIZE MAX_FRAMES PREFIX_CACHE
METHODS="${METHODS:-e0,e1,e2,e3,e4_s4,e4_s2,threshold5,stride32_e0,direct_generated,direct_likelihood_stride16,direct_promptfix,direct_stride32,direct_stride32_kvcache,caption_controlled}"

[[ "$(wc -l < "${INDEX_FILE}")" -eq 360 ]] || { echo "MSAD index is not 360 lines: ${INDEX_FILE}" >&2; exit 2; }
[[ "$(wc -l < "${TEMPORAL_ANNOTATION_FILE}")" -eq 360 ]] || { echo "MSAD temporal annotation is not 360 lines: ${TEMPORAL_ANNOTATION_FILE}" >&2; exit 2; }

GPU_COUNT="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
if ! "${PYTHON_BIN}" - "${GPU_COUNT}" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
available = torch.cuda.is_available()
count = torch.cuda.device_count()
if not available or count < expected:
    print(
        f"CUDA preflight failed: available={available}, visible_devices={count}, "
        f"required_workers={expected}. Check nvidia-smi, the NVIDIA driver, "
        "and the active VAA environment before starting inference.",
        file=sys.stderr,
    )
    raise SystemExit(3)
print(f"CUDA preflight OK: {count} visible device(s); launching {expected} worker(s).")
PY
then
  exit 3
fi

contains() { [[ ",${METHODS}," == *",$1,"* ]]; }

run_cumulative() {
  local output="$1"
  echo "[MSAD-360] cumulative: ${output}"
  DATASET_DIR="${DATASET_DIR}" VIDEO_DIR="${VIDEO_DIR}" INDEX_FILE="${INDEX_FILE}" \
  EVAL_ANNOTATION_FILE="${INDEX_FILE}" TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE}" \
  ROOT_PATH="${ROOT_PATH}" OUTPUT_DIR="${output}" MODEL_PATH="${MODEL_PATH}" \
  GPU_IDS="${GPU_IDS}" NORMAL_LABEL="${NORMAL_LABEL}" PRECISE_TIME=1 \
  FRAME_INTERVAL=16 WINDOW_SECONDS=10 SAMPLE_FPS=2 MAX_FRAMES="${MAX_FRAMES}" \
  THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE}" PREFIX_CACHE="${PREFIX_CACHE}" SAVE_THRESHOLD_DETAILS=0 \
  bash scripts/run_video_cumulative_rebalanced.sh
}

run_ab() {
  local experiment="$1" output="$2"
  echo "[MSAD-360] ${experiment}: ${output}"
  DATASET_DIR="${DATASET_DIR}" VIDEO_DIR="${VIDEO_DIR}" INDEX_FILE="${INDEX_FILE}" \
  EVAL_ANNOTATION_FILE="${INDEX_FILE}" TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE}" \
  ROOT_PATH="${ROOT_PATH}" OUTPUT_DIR="${output}" MODEL_PATH="${MODEL_PATH}" \
  GPU_IDS="${GPU_IDS}" NORMAL_LABEL="${NORMAL_LABEL}" PRECISE_TIME=1 \
  EXPERIMENT="${experiment}" bash scripts/run_video_ab_calibrated_rebalanced.sh
}

run_e3() {
  local output="$1"
  echo "[MSAD-360] E3: ${output}"
  DATASET_DIR="${DATASET_DIR}" VIDEO_DIR="${VIDEO_DIR}" INDEX_FILE="${INDEX_FILE}" \
  EVAL_ANNOTATION_FILE="${INDEX_FILE}" TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE}" \
  ROOT_PATH="${ROOT_PATH}" OUTPUT_DIR="${output}" MODEL_PATH="${MODEL_PATH}" \
  GPU_IDS="${GPU_IDS}" NORMAL_LABEL="${NORMAL_LABEL}" PRECISE_TIME=1 \
  bash scripts/run_video_yesno_calibrated_rebalanced.sh
}

run_e4() {
  local seconds="$1" output="$2"
  echo "[MSAD-360] E4 center=${seconds}s: ${output}"
  DATASET_DIR="${DATASET_DIR}" VIDEO_DIR="${VIDEO_DIR}" INDEX_FILE="${INDEX_FILE}" \
  EVAL_ANNOTATION_FILE="${INDEX_FILE}" TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE}" \
  ROOT_PATH="${ROOT_PATH}" OUTPUT_DIR="${output}" MODEL_PATH="${MODEL_PATH}" \
  GPU_IDS="${GPU_IDS}" NORMAL_LABEL="${NORMAL_LABEL}" PRECISE_TIME=1 \
  CENTER_SECONDS="${seconds}" bash scripts/run_video_center_dense_rebalanced.sh
}

run_direct() {
  local mode="$1" interval="$2" output="$3"
  echo "[MSAD-360] direct mode=${mode}, interval=${interval}: ${output}"
  DATASET_DIR="${DATASET_DIR}" VIDEO_DIR="${VIDEO_DIR}" INDEX_FILE="${INDEX_FILE}" \
  EVAL_ANNOTATION_FILE="${INDEX_FILE}" TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE}" \
  ROOT_PATH="${ROOT_PATH}" OUTPUT_DIR="${output}" MODEL_PATH="${MODEL_PATH}" \
  GPU_IDS="${GPU_IDS}" NORMAL_LABEL="${NORMAL_LABEL}" PRECISE_TIME=1 \
  FRAME_INTERVAL="${interval}" OUTPUT_INTERVAL=16 SCORE_MODE="${mode}" \
  bash scripts/run_video_direct_likelihood_rebalanced.sh
}

run_stride32_e0() {
  local output="${DATASET_DIR}/scores/videollama3_cumulative_likelihood_stride32_interp16"
  echo "[MSAD-360] E0 stride32->16 interpolation: ${output}"
  DATASET_DIR="${DATASET_DIR}" VIDEO_DIR="${VIDEO_DIR}" INDEX_FILE="${INDEX_FILE}" \
  EVAL_ANNOTATION_FILE="${INDEX_FILE}" TEMPORAL_ANNOTATION_FILE="${TEMPORAL_ANNOTATION_FILE}" \
  ROOT_PATH="${ROOT_PATH}" OUTPUT_DIR="${output}" MODEL_PATH="${MODEL_PATH}" \
  GPU_IDS="${GPU_IDS}" NORMAL_LABEL="${NORMAL_LABEL}" PRECISE_TIME=1 \
  FRAME_INTERVAL=32 OUTPUT_INTERVAL=16 WINDOW_SECONDS=10 SAMPLE_FPS=2 MAX_FRAMES="${MAX_FRAMES}" \
  THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE}" PREFIX_CACHE="${PREFIX_CACHE}" SAVE_THRESHOLD_DETAILS=0 \
  bash scripts/run_video_cumulative_stride32_rebalanced.sh
}

run_threshold5() {
  local output="${DATASET_DIR}/scores/videollama3_cumulative_likelihood_threshold5_actual_stride16"
  IFS=',' read -r -a gpus <<< "${GPU_IDS}"
  local jobs="${#gpus[@]}"; mkdir -p "${output}"
  echo "[MSAD-360] threshold5: ${output} (${jobs} workers)"
  local pids=()
  for job in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpus[$job]}" "${PYTHON_BIN}" -u -m src.video_cumulative_threshold5 \
      --video_dir "${VIDEO_DIR}" --index_file "${INDEX_FILE}" --output_dir "${output}" \
      --model_path "${MODEL_PATH}" --device cuda:0 --thresholds 0.2,0.4,0.6,0.8,1.0 \
      --threshold_batch_size "${THRESHOLD_BATCH_SIZE}" --max_frames "${MAX_FRAMES}" \
      --checkpoint_interval 20 --precise_time \
      --num_jobs "${jobs}" --job_index "${job}" --save_threshold_details --resume &
    pids+=("$!")
  done
  wait "${pids[@]}"
  "${PYTHON_BIN}" -u -m src.eval --root_path "${ROOT_PATH}" --annotationfile_path "${INDEX_FILE}" \
    --temporal_annotation_file "${TEMPORAL_ANNOTATION_FILE}" --scores_dir "${output}" \
    --output_dir "${output}/metrics_360" --frame_interval 16 --normal_label "${NORMAL_LABEL}" \
    --smoothing_sigma 10
}

run_caption_controlled() {
  local output="${DATASET_DIR}/scores/caption_e0_controlled_videollama3_stride16"
  local captions="${DATASET_DIR}/captions/video_llama3_json_results"
  IFS=',' read -r -a gpus <<< "${GPU_IDS}"
  local jobs="${#gpus[@]}"; mkdir -p "${output}"
  echo "[MSAD-360] caption-controlled E0: ${output} (${jobs} workers)"
  local pids=()
  for job in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpus[$job]}" "${PYTHON_BIN}" -u -m src.caption_e0_controlled \
      --index_file "${INDEX_FILE}" --captions_dir "${captions}" --output_dir "${output}" \
      --model_path "${MODEL_PATH}" --device cuda:0 --likelihood_temperature 1.0 \
      --threshold_batch_size "${THRESHOLD_BATCH_SIZE}" --checkpoint_interval 20 --num_jobs "${jobs}" \
      --job_index "${job}" --resume --progress_label "MSAD Caption-E0" &
    pids+=("$!")
  done
  wait "${pids[@]}"
  "${PYTHON_BIN}" -u -m src.eval --root_path "${ROOT_PATH}" --annotationfile_path "${INDEX_FILE}" \
    --temporal_annotation_file "${TEMPORAL_ANNOTATION_FILE}" --scores_dir "${output}" \
    --output_dir "${output}/metrics_360" --frame_interval 16 --normal_label "${NORMAL_LABEL}" \
    --smoothing_sigma 10
}

TOTAL_STAGES=0
contains e0 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains e1 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains e2 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains e3 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains e4_s4 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains e4_s2 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains threshold5 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains stride32_e0 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains direct_generated && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains direct_likelihood_stride16 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains direct_promptfix && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains direct_stride32 && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains direct_stride32_kvcache && TOTAL_STAGES=$((TOTAL_STAGES + 1))
contains caption_controlled && TOTAL_STAGES=$((TOTAL_STAGES + 1))

CURRENT_STAGE=0
count_videos() {
  local output="$1"
  find "${output}" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' '
}

print_progress() {
  local name="$1" output="$2" done total=360 width=32 filled empty percent
  done="$(count_videos "${output}")"
  ((done > total)) && done="${total}"
  filled=$((done * width / total))
  empty=$((width - filled))
  printf -v bar '%*s' "${filled}" ''
  bar="${bar// /#}"
  printf -v spaces '%*s' "${empty}" ''
  bar="${bar}${spaces}"
  percent=$((done * 100 / total))
  printf '\r[%d/%d] %-28s [%s] %3d%% (%d/360 videos)' \
    "${CURRENT_STAGE}" "${TOTAL_STAGES}" "${name}" "${bar}" "${percent}" "${done}"
}

run_stage() {
  local name="$1" output="$2" log_name="$3"
  shift 3
  CURRENT_STAGE=$((CURRENT_STAGE + 1))
  echo
  echo "===== [${CURRENT_STAGE}/${TOTAL_STAGES}] ${name} ====="
  echo "output=${output}"
  local attempt=1 max_attempts=3 status=0
  local stage_log="/tmp/covas_msad_${CURRENT_STAGE}_${log_name}.log"
  while ((attempt <= max_attempts)); do
    if ((attempt > 1)); then
      echo "显存保护重试 ${attempt}/${max_attempts}: batch=${THRESHOLD_BATCH_SIZE}, max_frames=${MAX_FRAMES}（保持不变）, prefix_cache=${PREFIX_CACHE}"
    fi
    # Child scripts are quiet; this wrapper owns the terminal progress bar.
    : >"${stage_log}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}" \
      TOKENIZERS_PARALLELISM=false "$@" >"${stage_log}" 2>&1 &
    local pid=$!
    local monitor_pid
    monitor_memory "${pid}" "${name}" &
    monitor_pid=$!
    while kill -0 "${pid}" 2>/dev/null; do
      print_progress "${name}" "${output}"
      sleep 5
    done
    status=0
    wait "${pid}" || status=$?
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    print_progress "${name}" "${output}"
    echo
    if [[ "${status}" -eq 0 ]]; then
      echo "完成：${name}"
      return 0
    fi
    if rg -qi 'out of memory|cuda out of memory|cuda error|cublas.*alloc|oom' "${stage_log}"; then
      echo "检测到 CUDA 显存不足，准备自动降级并重试；错误摘要：" >&2
      rg -i 'out of memory|cuda out of memory|cuda error|cublas.*alloc|oom' "${stage_log}" | tail -3 >&2 || true
      THRESHOLD_BATCH_SIZE=$((THRESHOLD_BATCH_SIZE > 2 ? THRESHOLD_BATCH_SIZE / 2 : 2))
      # Keep prefix/KV cache enabled and never alter MAX_FRAMES: both affect
      # the intended experiment.  Only text threshold batching is reduced;
      # each scorer clears unused CUDA allocator blocks at checkpoints.
      export THRESHOLD_BATCH_SIZE MAX_FRAMES PREFIX_CACHE
      attempt=$((attempt + 1))
      continue
    fi
    echo "任务失败（返回码 ${status}）；错误日志：${stage_log}" >&2
    tail -40 "${stage_log}" >&2 || true
    return "${status}"
  done
  echo "显存保护重试 ${max_attempts} 次仍失败；错误日志：${stage_log}" >&2
  return 1
}

monitor_memory() {
  local pid="$1" name="$2" warned=""
  while kill -0 "${pid}" 2>/dev/null; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      while IFS=',' read -r index used total; do
        index="${index// /}"; used="${used// /}"; total="${total// /}"
        [[ "${total}" =~ ^[0-9]+$ && "${used}" =~ ^[0-9]+$ && "${total}" -gt 0 ]] || continue
        local percent=$((used * 100 / total))
        if ((percent >= 90)); then
          if [[ "${warned}" != *",${index},"* ]]; then
            printf '\n显存警告：%s 使用 GPU %s 达到 %s%% (%s/%s MiB)；若 OOM 将自动降级。\n' \
              "${name}" "${index}" "${percent}" "${used}" "${total}" >&2
            warned="${warned},${index},"
          fi
        fi
      done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    fi
    sleep 5
  done
}

contains e0 && run_stage "E0 十阈值累计 likelihood" \
  "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_optimized" e0 \
  run_cumulative "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_optimized"
contains e1 && run_stage "E1 单向 A/B 校准" \
  "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e1_ab" e1 \
  run_ab e1 "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e1_ab"
contains e2 && run_stage "E2 A/B 交换平均校准" \
  "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e2_ab_swap" e2 \
  run_ab e2 "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e2_ab_swap"
contains e3 && run_stage "E3 YES/NO 极性反转校准" \
  "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e3_yesno_polarity_swap" e3 \
  run_e3 "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e3_yesno_polarity_swap"
contains e4_s4 && run_stage "E4 中心 4 秒密集采样" \
  "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e4_center_dense_g4_c6_s4" e4_s4 \
  run_e4 4 "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e4_center_dense_g4_c6_s4"
contains e4_s2 && run_stage "E4 中心 2 秒密集采样" \
  "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e4_center_dense_g4_c6_s2" e4_s2 \
  run_e4 2 "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_e4_center_dense_g4_c6_s2"
contains threshold5 && run_stage "五阈值累计 likelihood" \
  "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_threshold5_actual_stride16" threshold5 \
  run_threshold5
contains stride32_e0 && run_stage "Stride32 E0 插值回 Stride16" \
  "${DATASET_DIR}/scores/videollama3_cumulative_likelihood_stride32_interp16" stride32_e0 \
  run_stride32_e0
contains direct_generated && run_stage "直接生成式异常分数" \
  "${DATASET_DIR}/scores/videollama3_direct_generated_stride16" direct_generated \
  run_direct generated 16 "${DATASET_DIR}/scores/videollama3_direct_generated_stride16"
contains direct_likelihood_stride16 && run_stage "直接 11 类 likelihood Stride16" \
  "${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_optimized_stride16" direct_likelihood_stride16 \
  run_direct likelihood_optimized 16 "${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_optimized_stride16"
contains direct_promptfix && run_stage "直接 11 类 likelihood Stride16" \
  "${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_optimized_stride16_promptfix" direct_promptfix \
  run_direct likelihood_optimized 16 "${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_optimized_stride16_promptfix"
contains direct_stride32 && run_stage "直接 11 类 likelihood Stride32" \
  "${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_stride32_interp16" direct_stride32 \
  run_direct likelihood_optimized 32 "${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_stride32_interp16"
contains direct_stride32_kvcache && run_stage "直接 11 类 likelihood Stride32 KV-cache" \
  "${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_stride32_kvcache_interp16" direct_stride32_kvcache \
  run_direct likelihood_optimized 32 "${DATASET_DIR}/scores/videollama3_direct_11class_likelihood_stride32_kvcache_interp16"
contains caption_controlled && run_stage "Caption-E0 受控对照" \
  "${DATASET_DIR}/scores/caption_e0_controlled_videollama3_stride16" caption_controlled \
  run_caption_controlled

echo "MSAD 360 COVAS resume stages completed. Run scripts/eval_msad_360_all_existing.sh for the consolidated table."
