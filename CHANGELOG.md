# Changelog

## Unreleased

- Added English-only documentation for the MSAD 360-video protocol, prompt
  sensitivity, offline E0 diagnostics, and the complete ablation inventory.
- Clarified that UBnormal and unrelated exploratory branches are excluded from
  this release.
- Removed references to external VAD comparison data from the experiment
  documentation.

## 1.0.2 - 2026-08-03

- Added the research-code archive for E0, likelihood/generation comparisons,
  Caption-E0, E1-E4, stride/threshold/frame-order ablations, selective checks,
  SDEE pilot code, and legacy baseline entry points.
- Added preserved experiment score JSON files, metrics, timing/threshold
  metadata, partial-checkpoint records, and `results/EXPERIMENT_MANIFEST.json`.
- Added an experiment index with explicit complete/partial status and
  author-reported URF comparison values.

## 1.0.1 - 2026-07-31

- Included complete E0 score JSON files, metrics, evaluation annotations, and
  UCF-Crime per-threshold diagnostics for reproducibility.
- Added one-command evaluation for all bundled results.

## 1.0.0 - 2026-07-31

- Initial standalone open-source release of COVAS-VAD E0.
- Added direct VideoLLaMA3 cumulative ordinal scoring.
- Added shared visual encoding, threshold batching, and prefix KV caching.
- Added PAVA monotonic projection and continuous tail-probability scoring.
- Added atomic checkpoints, resumable multi-GPU scheduling, and evaluation.
- Added packaging metadata, configuration templates, documentation, and tests.
