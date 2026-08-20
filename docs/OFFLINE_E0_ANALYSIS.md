# Offline E0 Analysis

This report describes analyses that read existing E0 score files and do not
run VideoLLaMA3 inference. They are CPU-only and do not modify the stored
score JSON files.

## Completed analyses

- Gaussian smoothing sensitivity with sigma values 0, 2, 5, 10, and 20.
- Overlap mean, overlap max, and top-3 overlap aggregation on UCF-Crime.
- Entropy, residual entropy, and PAVA diagnostics.
- Selective center-4-second verification.
- Initial video-level bootstrap confidence intervals.
- PAVA on/off comparison for the complete UCF-Crime E0 output.

## Gaussian smoothing sensitivity

Metrics are ROC-AUC / PR-AUC / Max-F1.

| Dataset | sigma=0 | sigma=2 | sigma=5 | sigma=10 | sigma=20 |
|---|---:|---:|---:|---:|---:|
| UCF-Crime | 84.8153 / 37.6369 / 43.4400 | 85.7311 / 39.8825 / 45.3063 | 86.1208 / 40.6049 / 45.6038 | 86.2693 / 39.4706 / 45.2153 | 85.9467 / 36.6442 / 43.9617 |
| MSAD | 94.2910 / 81.9765 / 77.1855 | 94.5373 / 82.6563 / 77.3933 | 94.5866 / 82.4593 / 76.7592 | 94.3571 / 81.1967 / 76.2664 | 93.8952 / 79.1593 / 75.6210 |
| XD-Violence | 91.4096 / 74.5722 / 72.3337 | 92.0382 / 76.1349 / 73.4700 | 92.2261 / 76.4149 / 73.5152 | 92.1099 / 75.3590 / 73.1337 | 91.6475 / 73.2055 / 71.8817 |

The released E0 configuration keeps sigma=10 as a common fixed setting. The
sigma=5 result must not be promoted as a new default based only on test-set
metrics.

## PAVA diagnostic

For the complete UCF-Crime E0 result, PAVA on and off produce the same final
metrics:

```text
without PAVA: ROC 0.8626927 / PR 0.3947061 / Max-F1 0.4521527
with PAVA:    ROC 0.8626927 / PR 0.3947061 / Max-F1 0.4521527
```

This is expected because the final E0 score is the mean of the ten tail
probabilities, while isotonic projection preserves their sum. PAVA remains
useful as a probabilistic consistency constraint and for threshold-curve
diagnostics, but it should not be described as the source of this metric gain.

## Remaining limitations

Single-threshold and weighted ordinal-curve analyses require raw per-threshold
details. They cannot be reliably reconstructed from final `{center: score}`
JSON files because the ten threshold values have already been averaged.
