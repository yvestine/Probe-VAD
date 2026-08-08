# COVAS-VAD 实验归档与复现实验说明

本文档是 `COVAS-VAD` 中实验材料的总索引。它把主方法、作者对比、模型消融、
时间采样消融和历史探索分开记录，避免把未完成断点结果误报为完整测试结果。
所有百分比均为百分比点；ROC-AUC、PR-AUC 和 Max-F1 使用同一套评测接口。

## 1. 主方法与作者对比

主方法是 E0：VideoLLaMA3 直接读取原始视频 clip，针对
`0.1, ..., 1.0` 十个有序严重度阈值读取 YES/NO 条件 likelihood，使用 PAVA
单调投影后对十个尾概率求均值。默认输入为 10 秒窗口、16 帧中心间隔、2 FPS、
最多 10 帧，评测使用 Gaussian smoothing（sigma=10）。

| 数据集 | VADTree（作者） | URF-HVAA（作者） | COVAS-VAD E0 |
|---|---:|---:|---:|
| UCF-Crime | 84.74 / 41.89 / 44.29 | 84.36 / 36.15 / 41.83 | **86.27 / 39.47 / 45.22** |
| MSAD | 89.32 / 71.41 / 68.80 | 93.06 / 77.81 / 74.82 | **94.36 / 81.20 / 76.27** |
| XD-Violence | 90.47 / 67.91 / 69.17 | 91.34 / 68.07 / 71.93 | **92.11 / 75.36 / 73.13** |

表中每个单元格依次为 `ROC-AUC / PR-AUC / Max-F1`。作者数字来自论文/作者
公开结果；本仓库不重新实现作者方法，也不把 URF/VADTree 的内部中间分数打包为
COVAS-VAD 方法结果。

完整 E0 文件位置：

```text
results/<dataset>/scores/
results/<dataset>/metrics/
results/MANIFEST.json
```

## 2. 已完成的模型和输入消融

### 2.1 直接生成分数 vs. likelihood

两者使用原始视频 clip。Generation 要求输出 `[0.0]`–`[1.0]`；11-class
likelihood 对 11 个候选答案求条件概率期望；E0 则使用十个有序累计阈值。

| 数据集 | Generation | 11-class likelihood（promptfix） | E0 |
|---|---:|---:|---:|
| UCF-Crime | 70.20 / 25.73 / 30.56 | 84.43 / 40.25 / 44.75 | **86.27 / 39.47 / 45.22** |
| MSAD | 85.72 / 67.25 / 60.95 | 92.44 / 79.19 / 75.13 | **94.36 / 81.20 / 76.27** |
| XD-Violence | 86.18 / 68.78 / 71.26 | 91.77 / 77.15 / 74.64 | **92.11 / 75.36 / 73.13** |

Generation 的输出解析失败率为 0，但分数只有约 9–11 个离散值；likelihood
保留了几乎每个 clip 的连续排序信息。原始 score JSON 和 metrics 位于各数据集
的 `results/experiments/<dataset>/direct_generation`、
`direct_11class_likelihood_promptfix`。

### 2.2 原始视频 vs. Caption-E0

Caption-E0 保持累计 likelihood 的聚合思想，但将 clip 先压缩成 caption，再由
Llama 3.1 判断。它用于检验 caption 信息瓶颈，不是 COVAS-VAD 主流程。

| 数据集 | Video-E0 | Caption-E0 |
|---|---:|---:|
| UCF-Crime | **86.27 / 39.47 / 45.22** | 80.28 / 32.55 / 39.02 |
| MSAD | **94.36 / 81.20 / 76.27** | 92.14 / 78.82 / 75.24 |
| XD-Violence | **92.11 / 75.36 / 73.13** | 91.99 / 74.62 / 71.78 |

注意：当前 Caption-E0 使用 Llama 3.1 和不同的文本 prompt，因此它是完整
管线对照而不是严格的“只改变输入模态”单变量实验。若论文需要严格因果结论，
应使用同一 checkpoint 和匹配 prompt 重跑。

### 2.3 E1/E2/E3 选项校准（MSAD）

| 版本 | 改动 | ROC-AUC | PR-AUC | Max-F1 |
|---|---|---:|---:|---:|
| E0 | YES/NO 累计阈值 | **94.3630** | **81.2026** | **76.2655** |
| E1 | 单向 A/B | 93.2254 | 78.9846 | 75.6784 |
| E2 | A/B 正反语义 margin 平均 | 93.4871 | 79.4267 | 75.7109 |
| E3 | YES/NO 正反命题 margin 平均 | 93.9734 | 80.3436 | 76.1857 |

E2/E3 都严格执行“先平均 margin，再 sigmoid”，没有平均两次概率；20 个问题
共享视频解码和视觉编码。结果说明交换校准可以减轻选项偏置，但当前模型中
自然语义 YES/NO 仍优于 A/B。

对应目录：`results/experiments/MSAD/e1_ab`、`e2_ab_swap`、
`e3_yesno_polarity_swap`。

### 2.4 时间步长与阈值数

Stride32 只把模型评分中心从 16 帧改为 32 帧，然后线性插值回 16 帧；其余
E0 参数不变。Threshold5 是离线使用 `0.2,0.4,0.6,0.8,1.0` 五个尾概率
重新聚合，不重新推理。

| 数据集 | E0 Stride16 | Stride32 + 插值 | Stride32 + 5阈值 |
|---|---:|---:|---:|
| UCF-Crime | 86.27 / 39.47 / 45.22 | 86.16 / 39.38 / 45.25 | 86.23 / 39.46 / 45.69 |
| MSAD | 94.36 / 81.20 / 76.27 | 94.19 / 80.66 / 76.55 | 94.19 / 80.53 / 76.43 |
| XD-Violence | 92.11 / 75.36 / 73.13 | 92.09 / 75.28 / 73.06 | 92.07 / 75.12 / 72.93 |

对应目录：`e0_stride32`、`e0_stride32_threshold5`。归档只保留最终 score JSON
和 metrics；运行参数已经记录在本文档和 manifest 中。

### 2.5 帧采样与时间顺序（UCF-Crime）

| 版本 | 改动 | ROC-AUC | PR-AUC | Max-F1 |
|---|---|---:|---:|---:|
| E0-Order | 原始 10 帧时间顺序 | **86.2693** | **39.4706** | **45.2153** |
| E0-Shuffle | 固定 seed=17 打乱同一组帧 | 86.1425 | 38.4937 | 44.3540 |
| E0-Single-frame | 只使用窗口中心帧 | 83.2048 | 28.8855 | 36.7180 |
| E4-S2 | 中心密集采样，中心 2 秒 | 94.2867 | 81.0542 | 76.1688 |
| E4-S4 | 中心密集采样，中心 4 秒 | 94.3493 | 81.2101 | 76.2402 |

E4-S2/S4 的 MSAD score JSON 和 metrics 已保存，但它们不是三数据集统一完成的
主结果。对应代码位于
`research_code/src/video_center_dense_score.py` 和 `research_code/src/video_temporal_order_ablation.py`。

## 3. 选择性核查、窗口后处理和历史 refinement

这些实验不改变 E0 的主结论，作为诊断和后处理探索保留：

- `e0_selective_center_verify`：只对视频内 85%–95% 分位的候选 clip，用中心
  4 秒重新评分；当前目录保存核查 score JSON，不作为三数据集完整指标。
- `temporal_fusion/`：保存 overlapping mean、adjacent mean、overlap logit、
  adaptive 等跨窗口后处理结果。它们不是 COVAS-VAD 主方法，使用时必须注明
  是后处理消融，不能当作无先验 E0。
- `legacy_refinement/`：保存早期“异常标签 + 二次评分”的 UCF 结果，以及
  gate=0.49–0.51 的记录。它们依赖标签/refinement，不属于当前免 caption、
  免标签主流程。
- `exploratory/`：保存 UCF 上较早的 caption sampling、category generalization、
  Modify2、多尺度 caption fusion、VLM dense 2s、zero-shot anticipation、
  hazard/survival gate 等 pilot 输出。这些输出没有被提升为 COVAS-VAD 主表，
  但为避免丢失已全部归档；若某个 pilot 没有独立 runner 或完整指标，manifest
  会明确标记为探索性记录。
- `research_code/analysis/` 还保存阈值5离线聚合、PAVA/熵不确定性分析和
  SDEE pilot 代码；UCF 的 20-video SDEE-D1 order/shuffle/no-align 原始 JSON
  保存在 `results/experiments/ucf_crime/e5_sdee_pilot_20`，manifest 标记为
  `partial`，不应与完整 E0 指标并列。

### 3.1 跨窗口后处理的已保存指标

以下均为开启 sigma=10 后的评测结果，格式为
`ROC-AUC / PR-AUC / Max-F1`。它们只改变 E0 clip 序列的离线投影，不重新调用
VideoLLaMA3：

| 数据集 | Adaptive | Adjacent mean | Overlap logit | Overlap mean |
|---|---:|---:|---:|---:|
| UCF-Crime | 86.2430 / 39.2502 / 45.2260 | 86.2681 / 39.4580 / 45.2041 | 86.2671 / 39.2579 / 45.1442 | 86.2559 / 39.2315 / 45.0517 |
| MSAD | 94.3586 / 81.1573 / 76.3525 | 94.3633 / 81.1971 / 76.2539 | 94.3205 / 81.0350 / 76.2617 | 94.3081 / 80.9324 / 76.3035 |
| XD-Violence | 92.1051 / 75.3260 / 73.1846 | 92.1068 / 75.3417 / 73.1281 | 92.0983 / 75.2292 / 73.1533 | 92.0797 / 75.1972 / 73.0561 |

原始未平滑指标也保存在各方法目录的 `metrics_raw/`，便于重新比较不同平滑
策略。由于这些后处理没有在所有设置上稳定超过 E0，当前主方法仍保持 E0。

## 4. 结果完整性和复现规则

`results/EXPERIMENT_MANIFEST.json` 对每个归档目录记录：

1. 数据集和实验名称；
2. 源结果目录与打包后的相对路径；
3. score JSON 数量；
4. 是否完成该数据集测试索引；
5. 已报告的 ROC/PR/Max-F1（若有）；
6. 是否为完整测试集结果，以及对应的 metrics 是否存在。

完整目录才可以直接用于论文表格；部分目录只用于代码审计或失败分析。所有
保留的 score JSON 都保留原文件名和中心帧 key，不做重编号、不覆盖 E0。

## 5. 从归档继续运行

研究阶段的入口脚本位于 `research_code/scripts/`。脚本默认沿用原仓库的
相对路径，因此推荐在原仓库或把 `research_code/src` 加入 `PYTHONPATH` 后运行。
模型权重、视频和帧不随项目分发；运行者只需按脚本中的 `DATASET_DIR`、
`VIDEO_DIR`、`INDEX_FILE`、`MODEL_PATH` 和 `GPU_IDS` 改成自己的路径。

主方法优先使用顶层的 `scripts/run_from_config.sh`；研究消融只在需要复现实验
时调用 `research_code/scripts/` 中对应入口。
