# Experiments and Reproducibility

This document indexes the experiments bundled with COVAS-VAD. The primary
method is E0; the remaining directories are retained for comparison, audit,
and reproduction. Experiment status is recorded explicitly in
`results/EXPERIMENT_MANIFEST.json`.

## Primary method

COVAS-VAD E0 reads raw video clips and asks ten ordered YES/NO threshold
questions at severities `0.1, ..., 1.0`. It applies PAVA to the tail
probabilities and averages them into a continuous anomaly score. The default
configuration uses a 10-second window, a 16-frame center stride, 2 FPS, and at
most 10 input frames.

## Main results

The bundled complete E0 results are:

| Dataset | Videos | Clips | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|---:|---:|
| UCF-Crime | 290 | 69,634 | 86.27% | 39.47% | 45.22% |
| MSAD | 240 | 9,250 | 94.36% | 81.20% | 76.27% |
| XD-Violence | 800 | 146,449 | 92.11% | 75.36% | 73.13% |

The exact score files and compact metrics are under `results/<dataset>/`.
They can be evaluated without a model or GPU with:

```bash
bash scripts/evaluate_precomputed.sh all
```

## Baseline and ablation archive

`results/experiments/` contains preserved score JSON files and metrics for:

- caption-based E0 and direct generation;
- direct 11-class likelihood scoring;
- stride, frame-order, threshold-count, and center-density ablations;
- temporal-window fusion and uncertainty diagnostics;
- selective-center and SDEE pilot experiments;
- historical caption/tag/refinement baselines.

The corresponding source and launchers are under `research_code/src/`,
`research_code/analysis/`, `research_code/scripts/`, and
`research_code/legacy_baseline/`. A directory marked `partial`, `pilot`, or
`diagnostic` must not be reported as a complete test-set result.

## URF-HVAA reference

The URF-HVAA values in the comparison table are reference values from the
upstream project. This repository does not claim to reimplement URF-HVAA.
No external VAD method or dataset is part of the COVAS-VAD experiment scope.
See [`docs/THREE_DATASET_RESULTS_COMPARISON.md`](THREE_DATASET_RESULTS_COMPARISON.md)
for the exact values and provenance.

## Additional English experiment reports

The current research update adds the following English-only reports:

- [`MSAD_360_EVALUATION.md`](MSAD_360_EVALUATION.md): commands and output
  conventions for the complete 360-video MSAD protocol.
- [`MSAD_PROMPT_SENSITIVITY.md`](MSAD_PROMPT_SENSITIVITY.md): controlled
  threshold-question prompt comparison on MSAD.
- [`ABLATION_RESULTS.md`](ABLATION_RESULTS.md): consolidated E0, sampling,
  threshold, likelihood, and temporal-fusion results.
- [`OFFLINE_E0_ANALYSIS.md`](OFFLINE_E0_ANALYSIS.md): smoothing, PAVA,
  uncertainty, bootstrap, and selective-verification diagnostics.

These reports contain no UBnormal experiments. UBnormal-specific scripts,
annotations, checkpoints, and results are intentionally excluded.

## Reproduction boundaries

The repository does not include raw videos, extracted frames, model weights,
Hugging Face caches, or temporary runtime diagnostics. Dataset-specific
annotations and final score JSON files are included for the bundled offline
evaluation, subject to the corresponding dataset licenses.
