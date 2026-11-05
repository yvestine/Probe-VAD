# Factorized Flat-Class Likelihood matched ablation

## Status — 2026-11-05
Local implementation based on upstream `5d705bb9b0f808012be690700b7b876cc7ceb03d`. No original methods, historical experiments or manuscript results changed. The ablation is distributed on a separate development branch.
**Real model smoke and full MSAD inference have NOT run.** This Mac has no CUDA; public `results/msad/annotations/` has only 240 videos. Supply the full 360-video index, temporal annotations and raw videos on the GPU server.

Local checks: 14 tests pass, including two generated videos with a deterministic stand-in model. Actual downloaded VideoLLaMA3-7B tokenizer: YES=[14004], NO=[8996], both one token (runtime also checks and records this). Latest CPU compatibility tests: Python 3.10.20, Torch 2.5.1, torchvision 0.20.1, Transformers 4.46.3, NumPy 1.25.2, SciPy 1.9.3, scikit-learn 1.1.3, Accelerate 0.26.0. This supersedes the initial newer-runtime checks. Real CUDA/model inference remains unverified locally.

## Motivation and limits
Flat vs ordinal changes both question organization and candidate semantics. Factorized Flat-11 retains the eleven exact score classes but asks eleven one-vs-rest binary questions. Factorized−Flat estimates the combined change from binary decomposition and its scoring interface. Probe−Factorized compares cumulative threshold judgments against exact-class judgments under binary answers.
This is improved control, not a perfect causal decomposition: numeric-token candidates become YES/NO, prompts change, and cross-question margin normalization differs from original candidate normalization. Do not claim a uniquely identified percentage of gain caused by decomposition. Negative or uncertain effects must be reported without tuning.

## Three methods
| Mode | Questions | Scoring |
|---|---:|---|
| flat_11class_matched | 1 | Original length-normalized likelihood_optimized for `[0.0]`…`[1.0]`; T=1; expected score |
| factorized_flat_11class | 11 | YES/NO exact-class judgments; softmax of margins; expected score; no PAVA |
| probe_vad_matched | 10 | Original cumulative 0.1…1.0 binary probes, original projection/integration |

Flat delegates to the same `video_score_utils.score_conversation(..., 'likelihood_optimized', ..., length_normalize=True, temperature=1)` used by `video_direct_score.py`. Probe delegates to the cumulative engine and scoring function used by `video_cumulative_score.py`. The original files remain unchanged. The new matched driver accepts `--prompt`; the shell explicitly passes CUMULATIVE_PROMPT for all three. Historical DIRECT_PROMPT results are not controls.

All three modes use the existing direct runner's `preload_video_conversation` path before scoring. This controls actual sampled frames (including timestamp repair), rather than assuming legacy direct and cumulative decoding always agree. Each window is decoded once and visually encoded once PER METHOD; frames/timestamps are hashed across separate runs and must match before evaluation. Only the wrapper's input path is harmonized; Probe scoring mathematics is unchanged.

## Complete prompts
Shared system prompt, imported from `video_cumulative_score.py`:
```
You are a careful video anomaly detector for public surveillance scenes. Judge only visible evidence in the current video segment. Use an ordered anomaly-severity scale from 0 to 1: 0 means ordinary safe activity; 0.5 means clearly concerning or plausibly anomalous activity; and 1 means unmistakably severe, dangerous, violent, criminal, or emergency activity. Intermediate thresholds preserve this order. Do not infer events that are not visually supported.
```
Original flat user prompt:
```
Return only one bracketed score selected from [0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]. Do not output any other text.
```
Factorized user prompt; SCORE takes each value 0.0…1.0:
```
Considering the allowed scores [0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], and [1.0], is [SCORE] the single most appropriate anomaly severity score for this video segment, based only on visible evidence? Answer exactly YES or NO.
```
Probe user prompt; THRESHOLD takes 0.1…1.0:
```
On the severity scale defined by the system, is the anomaly severity visible in this video segment at least THRESHOLD? Judge only the current video's visual evidence. Answer exactly YES or NO.
```

## Mathematics and continuation
For k=0,…,10:
- m_k = log P(YES | V,q_k) − log P(NO | V,q_k).
- w_k = exp(m_k−max(m)) / sum_j exp(m_j−max(m)), fixed T=1.
- S(V) = sum_k (k/10) w_k.

Weights are normalized candidate preferences, not calibrated class posteriors. No PAVA for factorized Flat.
Answers are encoded without special tokens, appended after the assistant generation prefix, with no EOS. Runtime saves `_tokenizer.json`. Single-token mode reuses original optimized next-token scoring with prefix cache. Original loops use len(conversations), not ten; tests cover 11 questions and batch size 4. Multi-token mode shares vision but teacher-forces and SUMS all candidate token log probabilities; no length averaging, first-token shortcut, or text prefix cache in this fallback.

## Fixed protocol
VideoLLaMA3-7B; 10-second windows; frame_interval=16; sample_fps=2; max_frames=10; precise_time=True; T=1. Only text batch size may be changed for memory.
Require 360 unique full videos starting at frame zero, 120 normal videos labeled 0 and 240 abnormal videos. Index and temporal annotation names must match. Actual frame counts must match the index. Existing path resolver, window generator, annotation parser, labels and Gaussian expansion are reused.
Sigma=10 in clip-score steps, then repeat each score 16 times, as in the existing evaluator. normal_label=0. No test-set prompt/temperature selection.

## Output/recovery
Repository-root `results/experiments/MSAD/`:
- `factorized_flat_11class/<video>.json`: frame-key → score in original format.
- `_details/<video>.json` inside each method: factorized candidates with candidate value, yes_loglik, no_loglik, margin, weight; final score; frame hash/timestamps; window bounds; elapsed time.
- `_errors/<video>.json`: failures; empty after successful retry. No placeholder scores.
- `flat_11class_matched/`, `probe_vad_matched/`: separate control directories.
- `_smoke_factorized_flat_11class/`: separate two-video smoke, all three methods.
- `_report.json`: completion, coverage, failure count, timings, full-only metrics and paired intervals.

Atomic writes, periodic checkpoints and finally-flush preserve completed work on KeyboardInterrupt. Resume requires both scores and details and matching provenance. Corrupt JSON fails rather than silently restarting. Errors are retried; missing/error/default-filled results cannot pass evaluation. Nonempty unowned outputs and mismatched config are rejected. Per-method file locking prevents concurrent writers; no multi-GPU sharding is implemented. Do not change source videos or a local model checkpoint in place during resume.

## NVIDIA environment — verified against GitHub 2026-11-05
Sources: [environment.yml](https://github.com/yvestine/Probe-VAD/blob/main/environment.yml), [requirements.txt](https://github.com/yvestine/Probe-VAD/blob/main/requirements.txt), [installation](https://github.com/yvestine/Probe-VAD#installation). Upstream HEAD is still `5d705bb9b0f808012be690700b7b876cc7ceb03d`.

Target: **Linux/NVIDIA, Python 3.10, Torch 2.5.1, torchvision 0.20.1, CUDA runtime 12.4, Transformers 4.46.3**. The original model loader uses BF16. A BF16-capable GPU such as A100 is required; actual hardware/driver support is tested by the kernel preflight.

Create a NEW environment on the GPU server:
```bash
bash research_code/scripts/setup_factorized_flat_cuda.sh
conda activate probe-vad-factorized
export PYTHON_BIN="$(command -v python)"
bash research_code/scripts/run_factorized_flat_11class_msad360.sh check
```
The installer consumes the unchanged upstream environment.yml and adds ffmpeg/ffprobe, ffmpeg-python, imageio, einops and test/build tools. Remote VideoLLaMA3 processor imports ffmpeg-python/imageio beyond the original dependency list. FlashAttention installation uses --no-deps to prevent replacing pinned Torch. Source builds require a compatible CUDA 12.4 toolkit/nvcc and C++ compiler; the runtime alone is not a compiler. Upstream does not pin a FlashAttention version, and no tested version is invented here.

For the repository-supported eager alternative, still on NVIDIA CUDA:
```bash
ATTN_IMPLEMENTATION=eager bash research_code/scripts/setup_factorized_flat_cuda.sh
conda activate probe-vad-factorized
export PYTHON_BIN="$(command -v python)"
export ATTN_IMPLEMENTATION=eager
bash research_code/scripts/run_factorized_flat_11class_msad360.sh check
```
If setup already created the environment before FlashAttention failed, do not repeat environment creation: activate it, set ATTN_IMPLEMENTATION=eager, then run check and tests. Eager can use more memory; lower THRESHOLD_BATCH_SIZE (down to 1), never max_frames. Keep the backend identical across methods and repeat smoke in a new results root when changing it.

Preflight checks pinned package versions, Linux/CUDA, CUDA build 12.4, FFmpeg tools, actual CUDA BF16 matrix multiplication and (when selected) a FlashAttention kernel. CPU/MPS fallback is forbidden in the production runner. Runtime packages and GPU metadata are written to `_runtime.json`. The Python CLI also runs preflight, so it cannot bypass these checks by omitting the shell launcher.

## GPU commands
From repository root, with the supported inference environment:
```bash
export PYTHON_BIN=/path/to/inference-env/bin/python
export VIDEO_DIR=/data/MSAD/videos
export INDEX_FILE=/data/MSAD/annotations/test_360.txt
export TEMPORAL_ANNOTATION_FILE=/data/MSAD/annotations/temporal_360.txt
export MODEL_PATH=DAMO-NLP-SG/VideoLLaMA3-7B
export CUDA_VISIBLE_DEVICES=0
bash research_code/scripts/run_factorized_flat_11class_msad360.sh smoke
bash research_code/scripts/run_factorized_flat_11class_msad360.sh full
```
Smoke uses the first two COMPLETE videos from the full index, validates all windows and cross-method frame hashes, and does not publish dataset metrics. Full mode requires smoke from the same code/model and revalidates its files. Repeat the same command to resume. Preflight rejects missing data, the public 240-video index, frame-count mismatches and unavailable CUDA.

## Evaluation and bootstrap
ROC-AUC, trapezoidal PR-AUC, sklearn AP and Max-F1 are distinct fields. PR-AUC is not relabeled AP. Max-F1 is a curve summary, not a deployment threshold.
Paired video-cluster bootstrap (seed 2026, >=1000 repeats) draws videos with replacement and pools ALL frames with the same draw across methods. Percentile 95% intervals: Factorized−Flat, Probe−Factorized, Probe−Flat for AUC and AP. Units are fractions; multiply by 100 for percentage points. Single-class draws are skipped; fewer than 1000 valid replicates aborts. Full-data bootstrapping can be CPU-intensive.
Elapsed-window totals include decoding/hashing/scoring but exclude model loading, failed attempts and interruption downtime; these are not total wall-clock measurements. No inference speedup has been measured.

| Method | Completed videos | ROC-AUC | AP | PR-AUC | Max-F1 | Time |
|---|---|---|---|---|---|---|
| Flat matched | Not run | Not measured | Not measured | Not measured | Not measured | Not measured |
| Factorized | Not run | Not measured | Not measured | Not measured | Not measured | Not measured |
| Probe matched | Not run | Not measured | Not measured | Not measured | Not measured | Not measured |

Real failures and paired confidence intervals are unknown, not zero. Injected synthetic failures only test retry behavior.

## Tests
`python -m pytest tests/test_factorized_flat.py -q` in the activated environment. Matching-version local check uses `.venv-py310/bin/python`.
14 cases cover question invariance, 11-query batching, complete multi-token sums, one visual encode/window, extreme margins, normalization/range, invalid likelihoods, ownership/config protection, strict completeness, interrupted checkpoint/resume and retry, identical-method bootstrap zero differences, and generated-video end-to-end smoke. This does not validate real VLM numerical equivalence, CUDA memory behavior or full-dataset performance.

The additional compatibility tests exercise actual Transformers 4.46.3 Llama/DynamicCache APIs: eleven cached and uncached continuation likelihoods agree within 2e-6 on a tiny CPU model. The CUDA package version contract is also tested. This is not a real GPU benchmark.
