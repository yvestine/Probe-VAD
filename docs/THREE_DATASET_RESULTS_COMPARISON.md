# 三个数据集实验结果对比

所有指标均为百分比（%），粗体表示对应数据集上的最优结果。

| 数据集 | 方法 | ROC-AUC ↑ | PR-AUC ↑ | Max-F1 ↑ |
|---|---|---:|---:|---:|
| UCF-Crime | VADTree | 84.74 | **41.89** | 44.29 |
| UCF-Crime | URF-HVAA | 84.36 | 36.15 | 41.83 |
| UCF-Crime | E0-Stride16 (Cumulative Likelihood) | **86.27** | 39.47 | 45.22 |
| UCF-Crime | E0-Stride32 + interpolation | 86.16 | 39.38 | **45.25** |
| MSAD | VADTree | 89.32 | 71.41 | 68.80 |
| MSAD | URF-HVAA | 93.06 | 77.81 | 74.82 |
| MSAD | E0-Stride16 (Cumulative Likelihood) | **94.36** | **81.20** | 76.27 |
| MSAD | E0-Stride32 + interpolation | 94.19 | 80.66 | **76.55** |
| XD-Violence | VADTree | 90.47 | 67.91 | 69.17 |
| XD-Violence | URF-HVAA | 91.34 | 68.07 | 71.93 |
| XD-Violence | E0-Stride16 (Cumulative Likelihood) | **92.11** | **75.36** | **73.13** |
| XD-Violence | E0-Stride32 + interpolation | 92.09 | 75.28 | 73.06 |
| 三数据集平均 | VADTree | 88.18 | 60.40 | 60.75 |
| 三数据集平均 | URF-HVAA | 89.59 | 60.68 | 62.86 |
| 三数据集平均 | E0-Stride16 (Cumulative Likelihood) | **90.91** | **65.34** | 64.87 |
| 三数据集平均 | E0-Stride32 + interpolation | 90.81 | 65.11 | **64.95** |

## 说明

- UCF-Crime：290/290 个测试视频，69,634/69,634 个 clips。
- MSAD：240/240 个测试视频，9,250/9,250 个 clips。d
- XD-Violence：800/800 个测试视频，146,449/146,449 个 clips。
- PR-AUC 使用 precision-recall 曲线的梯形积分计算，而不是 `average_precision_score`。
- E0-Stride16 是原始累计阈值似然流程；E0-Stride32 将模型评分中心间隔从 16 帧改为 32 帧，再线性插值回 16 帧输出分辨率。
- 两个 E0 版本均不生成 caption 或异常标签，直接通过累计阈值似然获得视频片段异常分数。
