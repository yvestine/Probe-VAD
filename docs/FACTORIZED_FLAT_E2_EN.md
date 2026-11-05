# Factorized Flat-Class Likelihood with E2 A/B Calibration

## Status and scope

This experiment replaces the earlier YES/NO Factorized Flat implementation. Its independent identifier is `factorized_flat_e2_11class`. The original E0, E1 and E2 implementations, shared inference utilities, evaluator and historical results are unchanged.

The implementation has passed 11 local CPU tests, including a two-video synthetic pipeline smoke test and a tiny Qwen2 cached/uncached likelihood comparison under Transformers 4.46.3. The synthetic models are test fixtures, not VideoLLaMA3 inference results. Real VideoLLaMA3 GPU smoke testing and the full MSAD-360 experiment have **not** been run. No dataset metrics or inference failure counts are claimed.

## Research question

Does replacing E2's cumulative threshold judgments with exact-class judgments change anomaly ranking when the A/B option-swap calibration mechanism is retained?

The experiment uses eleven exact scores, `[0.0]` through `[1.0]`, instead of ten cumulative thresholds. For each score it evaluates both assignments of A/B meanings, producing 22 prompts per video window.

Forward prompt for a candidate `[SCORE]`:

```text
Is [SCORE] the single most appropriate anomaly severity score for this video segment, based only on visible evidence?
A: [SCORE] is the single most appropriate anomaly severity score.
B: [SCORE] is not the single most appropriate anomaly severity score.
Answer exactly A or B.
```

The swapped prompt keeps the question and A/B order fixed but exchanges the two option meanings. The system prompt is imported unchanged from E2's `CUMULATIVE_PROMPT`:

```text
You are a careful video anomaly detector for public surveillance scenes. Judge only visible evidence in the current video segment. Use an ordered anomaly-severity scale from 0 to 1: 0 means ordinary safe activity; 0.5 means clearly concerning or plausibly anomalous activity; and 1 means unmistakably severe, dangerous, violent, criminal, or emergency activity. Intermediate thresholds preserve this order. Do not infer events that are not visually supported.
```

## Scoring

For candidate score `s_k = k/10`, k=0,...,10:

```text
m1_k = log P(A | forward query) - log P(B | forward query)
m2_k = log P(B | swapped query) - log P(A | swapped query)
m_k  = (m1_k + m2_k) / 2
p_k  = exp(m_k - max(m)) / sum_j exp(m_j - max(m))
Score = sum_k s_k * p_k
```

Temperature is fixed at T=1.0. The weights are normalized class preferences, not calibrated class posterior probabilities. Neither PAVA nor cumulative integration is applied. E2's calibration function is reused, but its returned sigmoid probability is discarded: Softmax operates on the calibrated **margins**, not sigmoid outputs.

The original E2 optimized/reference likelihood functions are reused. All 22 prompts are passed together to the optimized engine, sharing one visual encoding and preserving text batching and prefix caching. A/B single-token validation is also reused. The actual downloaded VideoLLaMA3 tokenizer encoded A as `[32]` and B as `[33]`; the runtime checks again and rejects unsupported tokenizations.

## Files

| File | Purpose |
|---|---|
| `research_code/src/video_factorized_flat_e2_score.py` | Independent scorer, diagnostics, checkpoint/resume and coverage checks |
| `research_code/scripts/run_factorized_flat_e2_msad360.sh` | GPU smoke/full launch and original evaluation |
| `tests/test_factorized_flat_e2.py` | Self-contained regression and synthetic smoke tests |
| `docs/FACTORIZED_FLAT_E2_EN.md` | This document |

No imports or runtime dependencies on the removed YES/NO experiment remain.

## Environment and fixed protocol

Use the original E2 NVIDIA environment. The repository specifies Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.4 and Transformers 4.46.3. The original loader uses BF16 on CUDA, so a compatible NVIDIA GPU is required.

For a new environment, follow the existing repository setup:

```bash
conda env create -f environment.yml
conda activate probe-vad
python -m pip install -e . --no-deps
python -m pip install pytest==8.3.5 ffmpeg-python imageio einops
```

Ensure `ffmpeg` and `ffprobe` executables are installed. For the default FlashAttention backend, install a build compatible with the pinned PyTorch/CUDA environment, as described in the repository README. Source compilation requires a compatible CUDA toolkit and compiler; a CUDA runtime alone is insufficient. The existing `eager` attention option is available via `ATTN_IMPLEMENTATION=eager`; use the same backend as the E2 control. Do not upgrade the pinned environment simply to resolve a FlashAttention installation issue.

Fixed settings:

- Model: `DAMO-NLP-SG/VideoLLaMA3-7B`.
- Window: 10 seconds; frame interval: 16; sampling: 2 FPS; maximum frames: 10.
- `precise_time=True`, T=1.0, original E2 video processing and model loading.
- Full MSAD protocol: 360 unique videos, including 120 normal videos labeled 0.
- Evaluation: original `src.eval`, frame interval 16, Gaussian sigma 10, normal label 0.

The Gaussian filter operates on the clip-score sequence before repeating scores for 16 frames. It does not use sigma 10 in original-video-frame units.

The index and temporal annotations must cover the same 360 videos. Videos must start at index frame 0, and actual frame counts must match the index. The public 240-video historical annotation files are **not** a substitute. Dataset videos and the complete 360-video annotation paths must be supplied on the execution server.

## Run

From the repository root, activate the E2 environment and replace the example paths with actual absolute paths:

```bash
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export VIDEO_DIR=/data/MSAD/videos
export INDEX_FILE=/data/MSAD/annotations/test_360.txt
export TEMPORAL_ANNOTATION_FILE=/data/MSAD/annotations/temporal_360.txt
export MODEL_PATH=DAMO-NLP-SG/VideoLLaMA3-7B

python -m pytest tests/test_factorized_flat_e2.py -q
bash research_code/scripts/run_factorized_flat_e2_msad360.sh smoke
bash research_code/scripts/run_factorized_flat_e2_msad360.sh full
```

For an offline model, use its complete local checkpoint path and the repository's offline settings. Do not enable offline mode before downloading the required weights and remote model code.

The smoke run processes all windows of the first two videos in the full index. It writes to a separate `factorized_flat_e2_11class_smoke` directory and does not publish dataset metrics. The launcher prechecks the full dataset paths and frame counts even in smoke mode. Full mode first validates a successful smoke with matching configuration/code, then runs all 360 videos, validates coverage, and calls the unchanged evaluator.

Optional execution settings:

```bash
export ATTN_IMPLEMENTATION=eager   # only if this also matches the E2 control
export THRESHOLD_BATCH_SIZE=2      # default 10; affects text batching only
export CHECKPOINT_INTERVAL=20
```

Do not reduce the visual frame budget or tune prompts/temperature on test results. Reuse existing E2 results only after confirming the same model, prompt, data split, video processing and evaluation protocol.

## Outputs and recovery

Default output root:

```text
results/experiments/MSAD/factorized_flat_e2_11class/
├── <video>.json
├── _calibration_details/<video>.json
├── _errors/<video>.json
├── _experiment.json
├── _tokenizer.json
├── _coverage.json
└── metrics_360/
```

Score files retain the original center-frame-to-score JSON format. Diagnostics store all eleven candidate scores; forward/swapped A/B log-likelihoods; both oriented margins; calibrated margins; normalized weights; final scores; and window boundaries. The inherited `*_logit` field names contain log-softmax values, whose differences equal raw-logit differences.

Atomic checkpointing and `finally` flushing preserve completed windows on KeyboardInterrupt. Repeat the same command to resume; failed or invalid windows are retried. No default scores are inserted for failures. Completed files are numerically ordered before the original evaluator reads them. Nonempty unowned directories and configuration mismatches are rejected. Use a new output directory when changing the experiment configuration; do not edit the provenance marker to bypass validation.

`_coverage.json` must report all 360 videos, every expected window, zero invalid/missing windows, zero failure records, and `complete=true` before evaluation. Do not infer completion from the number of JSON files alone. A failed rerun does not make old metric files current.

The unchanged evaluator reports ROC-AUC, trapezoidal PR-AUC and Max-F1. This minimal E2-based replacement does not add sklearn AP or bootstrap evaluation. Do not relabel PR-AUC as AP.

## Interpretation

The A/B likelihood and swap-calibration mechanism remains fixed, but the new experiment changes proposition semantics, candidate count and aggregation. Report the measured performance difference with those qualifications; it is not a fully isolated causal estimate of ordinality alone. No outcomes are currently available to support a performance claim.
