# Three-Dataset Results Comparison

All metrics are percentages. Bold values indicate the best result for each
dataset.

| Dataset | Method | ROC-AUC ↑ | PR-AUC ↑ | Max-F1 ↑ |
|---|---|---:|---:|---:|
| UCF-Crime | URF-HVAA | 84.36 | 36.15 | 41.83 |
| UCF-Crime | E0-Stride16 (Cumulative Likelihood) | **86.27** | 39.47 | 45.22 |
| UCF-Crime | E0-Stride32 + interpolation | 86.16 | 39.38 | **45.25** |
| MSAD | URF-HVAA | 93.06 | 77.81 | 74.82 |
| MSAD | E0-Stride16 (Cumulative Likelihood) | **94.36** | **81.20** | 76.27 |
| MSAD | E0-Stride32 + interpolation | 94.19 | 80.66 | **76.55** |
| XD-Violence | URF-HVAA | 91.34 | 68.07 | 71.93 |
| XD-Violence | E0-Stride16 (Cumulative Likelihood) | **92.11** | **75.36** | **73.13** |
| XD-Violence | E0-Stride32 + interpolation | 92.09 | 75.28 | 73.06 |
| Three-dataset average | URF-HVAA | 89.59 | 60.68 | 62.86 |
| Three-dataset average | E0-Stride16 (Cumulative Likelihood) | **90.91** | **65.34** | 64.87 |
| Three-dataset average | E0-Stride32 + interpolation | 90.81 | 65.11 | **64.95** |

## Notes

- UCF-Crime: 290/290 test videos and 69,634/69,634 clips.
- MSAD: 240/240 test videos and 9,250/9,250 clips.
- XD-Violence: 800/800 test videos and 146,449/146,449 clips.
- PR-AUC is computed by trapezoidal integration of the precision-recall
  curve rather than `average_precision_score`.
- E0-Stride16 is the original cumulative-likelihood pipeline. E0-Stride32
  changes the scoring-center interval from 16 to 32 frames and linearly
  interpolates scores back to the 16-frame output grid.
- Both E0 variants score raw video clips directly without generating captions
  or anomaly labels.
- This table intentionally excludes external VAD methods and UBnormal.
