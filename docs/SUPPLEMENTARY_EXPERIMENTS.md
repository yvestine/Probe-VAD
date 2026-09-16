
## Number of ordinal thresholds

We vary the number of cumulative severity thresholds while keeping the scoring rule unchanged.

| K | ROC-AUC (%) | PR-AUC (%) |
|---:|---:|---:|
| 1 | 86.97 | 76.77 |
| 2 | 87.15 | 77.51 |
| 5 | 87.41 | 78.12 |
| 10 (default) | **87.55** | **78.43** |

Increasing the number of thresholds improves performance, with diminishing returns once the severity axis is sufficiently populated. We retain K=10 as the default because it gives the best overall performance and provides a finer ordinal profile for structural analysis.

## Severity-threshold placement

To test sensitivity to threshold placement, we compare four five-threshold grids under the same K=5 budget.

| Grid | Thresholds | ROC-AUC (%) | PR-AUC (%) |
|---|---|---:|---:|
| Uniform | {0.2, 0.4, 0.6, 0.8, 1.0} | 87.41 | 78.12 |
| Low-focused | {0.1, 0.2, 0.3, 0.6, 1.0} | 87.28 | 77.76 |
| Mid-focused | {0.1, 0.3, 0.5, 0.7, 1.0} | 87.31 | 77.90 |
| High-focused | {0.1, 0.4, 0.7, 0.9, 1.0} | **87.49** | **78.15** |

The small variation indicates limited sensitivity to the exact threshold placement. We therefore retain the uniform grid as the default rather than selecting a test-set-specific alternative.

## Ordinal severity structure

The following analyses are performed offline on retained threshold-wise outputs. They require no additional VLM inference and are used only to characterize the internal ordinal response structure.

For threshold-wise diagnostics, the largest anomaly-normal mean separation occurs at severity 0.1 (difference 0.3294). The strongest standalone ranking occurs at severity 0.8 for ROC-AUC (87.54%) and at severity 0.4 for PR-AUC (78.39%). Thus, the threshold that maximizes average group separation does not necessarily provide the strongest sample-wise ranking.

For interval analysis, define

\[
d_{i,k}=\hat p_{i,k}-\hat p_{i,k+1},
\]

where \(\hat p_{i,k}\) denotes the PAVA-projected tail-evidence proxy at threshold \(\tau_k\). The quantity \(d_{i,k}\) is interpreted as an interval evidence transition rather than a calibrated probability mass.

| Severity band | Mean d(i,k) | Anomaly-normal gap | ROC-AUC (%) | PR-AUC (%) |
|---|---:|---:|---:|---:|
| 0.1-0.2 | 0.0080 | 0.0042 | 69.23 | 51.18 |
| 0.2-0.3 | 0.0089 | 0.0010 | 54.18 | 41.96 |
| 0.3-0.4 | 0.0410 | 0.0147 | 68.82 | 52.86 |
| 0.4-0.5 | 0.0318 | -0.0116 | 34.36 | 31.75 |
| 0.5-0.6 | **0.0529** | **0.0593** | 83.09 | 64.79 |
| 0.6-0.7 | 0.0115 | 0.0115 | 75.70 | 60.89 |
| 0.7-0.8 | 0.0200 | 0.0158 | 80.34 | 65.40 |
| 0.8-0.9 | 0.0248 | 0.0341 | **83.77** | **74.79** |
| 0.9-1.0 | 0.0238 | 0.0088 | 55.32 | 53.00 |

The 0.5-0.6 band shows the largest average transition and anomaly-normal gap, whereas 0.8-0.9 provides the strongest interval-level ranking. This supports aggregating the complete cumulative ordinal profile rather than relying on a single severity region.


## End-to-end efficiency

Efficiency is measured on a fixed balanced subset of 50 MSAD videos (25 anomalous and 25 normal), containing 2,235 clips and 22,350 sampled input frames. Effective FPS is computed from the sampled frames actually processed by the model.

| Pipeline | Wall-clock time | Effective FPS | Persistent storage |
|---|---:|---:|---:|
| COVAS-VAD (10 thresholds) | **1:58:37** | **3.14** | **0.77 MB** |
| Complete caption pipeline | 4:49:19 | 1.29 | 6.06 MB |

Relative to the complete caption pipeline, COVAS-VAD reduces wall-clock time by 59.0%, increases end-to-end effective throughput by 2.44x, and reduces persistent intermediate storage by 87.3%.

The storage comparison concerns persistent intermediate artifacts rather than the original visual input or transient runtime representations. Both pipelines use the same source videos and temporal sampling protocol, so raw videos are excluded.

## Default configuration retained

The supplementary experiments do not alter the released default configuration:

- 10 uniformly spaced cumulative severity thresholds;
- uniform clip-internal frame sampling;
- Gaussian smoothing with sigma=10;
- the same cumulative ordinal likelihood scoring rule as the main method.
