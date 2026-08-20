# COVAS-VAD Ablation Results

The following results use the complete 360-video MSAD protocol and share the
same evaluation annotations. Metrics are percentages reported as ROC-AUC /
PR-AUC / Max-F1.

| Experiment | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| E0 cumulative likelihood | 87.5544 | 78.4314 | 75.6379 |
| E1 one-way A/B calibration | 86.0590 | 76.9668 | 73.9566 |
| E2 swapped A/B averaging | 86.2121 | 77.2350 | 73.9013 |
| E3 YES/NO polarity reversal | 87.2652 | 78.3948 | 75.3290 |
| Center-dense sampling, 4 seconds | 87.5983 | 78.4507 | 75.6963 |
| Center-dense sampling, 2 seconds | 87.5364 | 78.3752 | 75.6678 |
| Five cumulative thresholds | 87.5135 | 78.3168 | 75.6624 |
| E0 stride 32 with interpolation | 87.4600 | 78.2452 | 75.7045 |
| Direct generated score | 78.2853 | 69.3340 | 64.5744 |
| Direct 11-class likelihood | 75.7882 | 68.3954 | 60.9953 |
| Direct 11-class likelihood, prompt fix | 86.7708 | 77.5636 | 73.7964 |
| Caption-E0 controlled comparison | 86.7912 | 78.3240 | 74.4712 |

## Interpretation

- Cumulative ordinal likelihood is substantially stronger than direct score
  generation and uncorrected 11-class likelihood.
- Reducing the threshold count to five causes only a small change on MSAD.
- Center-dense sampling produces a small improvement, but the gain is not
  large enough to replace the common E0 configuration without a fixed
  validation protocol.
- Stride 32 is a practical efficiency variant; interpolation preserves most
  of the E0 performance.
- Prompt and polarity variants are diagnostic ablations and should not be
  selected using the test-set metrics alone.
