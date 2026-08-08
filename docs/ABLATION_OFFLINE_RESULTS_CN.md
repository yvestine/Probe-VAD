# 离线消融实验结果

本文档记录当前已经生成的分数在不重新调用 VideoLLaMA3 的情况下完成的统一评测结果。

## 评测设置

- 评测脚本：`src/eval_interpolated.py`
- scoring interval：16 帧
- output interval：16 帧
- Gaussian smoothing：开启，`sigma=10`
- 分数展开：按中心帧排序后插值到 16 帧输出网格，再进行帧级评测
- UCF-Crime：290 个测试视频
- MSAD：使用 `anomaly_test.txt` 的 240 个视频进行公平比较
- XD-Violence：800 个测试视频

所有指标均为百分比。

## 1. Likelihood 与直接生成分数

两种方法均直接输入原始视频 clip。Likelihood 版本对 11 个候选分数计算条件 likelihood 并求期望；Generation 版本要求模型直接生成 `[0.0]` 到 `[1.0]` 中的一个分数。

| 数据集 | 直接 Generation ROC-AUC | 直接 Generation PR-AUC | 直接 Generation Max-F1 | Likelihood ROC-AUC | Likelihood PR-AUC | Likelihood Max-F1 |
|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime | 70.2006 | 25.7273 | 30.5620 | **84.4365** | **40.2451** | **44.8403** |
| MSAD | 85.7152 | 67.2532 | 60.9461 | **92.4355** | **79.1942** | **75.1304** |
| XD-Violence | 86.1817 | 68.7754 | 71.2585 | **91.7651** | **77.1522** | **74.6449** |

Likelihood 相对 Generation 的提升为：

- UCF：ROC +14.2359、PR +14.5178、Max-F1 +14.2783 个百分点；
- MSAD：ROC +6.7203、PR +11.9409、Max-F1 +14.1843 个百分点；
- XD：ROC +5.5834、PR +8.3768、Max-F1 +3.3864 个百分点。

### 分数解析与离散程度

以下统计均在 clip-level 分数上计算。Generation 的解析失败率来自各输出目录中的 `_errors` 记录；三个数据集均为 0。

| 数据集 | 方法 | clip 数 | 记录的解析失败率 | 唯一值数量 | 均值 ± 标准差 | 最大重复值占比 |
|---|---|---:|---:|---:|---:|---:|
| UCF-Crime | Generation | 69,634 | 0.0000% | 11 | 0.272567 ± 0.238710 | 46.7774% |
| UCF-Crime | Likelihood | 69,634 | 不适用 | 69,537 | 0.476320 ± 0.019350 | 0.0144% |
| MSAD | Generation | 9,250 | 0.0000% | 9 | 0.210768 ± 0.292181 | 47.4054% |
| MSAD | Likelihood | 9,250 | 不适用 | 9,122 | 0.473844 ± 0.036070 | 0.0757% |
| XD-Violence | Generation | 146,449 | 0.0000% | 11 | 0.475117 ± 0.285807 | 56.5105% |
| XD-Violence | Likelihood | 146,449 | 不适用 | 146,440 | 0.503349 ± 0.039566 | 0.0041% |

Generation 分数集中在少数离散值，尤其是 0.5；Likelihood 保留了几乎每个 clip 的连续排序信息。这与 ROC-AUC、PR-AUC 和 Max-F1 的差异一致。

## 2. 原始视频条件与 Caption 条件

两种方法均使用累计阈值 likelihood、PAVA 单调投影和等权积分。唯一变化是输入：

- Video-E0：VideoLLaMA3 直接观察原始视频 clip；
- Caption-E0：Llama 3.1 只观察预先生成的 clip caption。

| 数据集 | Video-E0 ROC-AUC | Video-E0 PR-AUC | Video-E0 Max-F1 | Caption-E0 ROC-AUC | Caption-E0 PR-AUC | Caption-E0 Max-F1 |
|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime | **86.2693** | **39.4706** | **45.2153** | 80.2765 | 32.5501 | 39.0150 |
| MSAD | **94.3630** | **81.2026** | **76.2655** | 92.1396 | 78.8225 | 75.2398 |
| XD-Violence | **92.1099** | **75.3590** | **73.1337** | 91.9873 | 74.6177 | 71.7832 |

Video-E0 相对 Caption-E0 的提升为：

- UCF：ROC +5.9928、PR +6.9205、Max-F1 +6.2003 个百分点；
- MSAD：ROC +2.2234、PR +2.3801、Max-F1 +1.0257 个百分点；
- XD：ROC +0.1226、PR +0.7413、Max-F1 +1.3505 个百分点。

这组结果支持“caption 可能造成信息瓶颈”的判断，但 XD 上两种输入的 ROC 差距较小，说明该数据集还需要结合类别级和短动作片段分析。

### 重要实验有效性说明

当前 Caption-E0 与 Video-E0 的评测和时间序列对齐是正确的，但二者并非严格的单变量消融：

- Video-E0 使用 VideoLLaMA3-7B；Caption-E0 使用本地 Llama 3.1-8B；
- Video-E0 的 system prompt 明确给出 0、0.5、1.0 的严重程度语义锚点，并要求判断视觉证据；
- Caption-E0 的 system prompt 改为判断 caption 描述的行为，没有完全保留上述 0.5 语义锚点；
- 两者的用户问题分别是“当前视频片段中可见证据是否达到阈值”和“caption 描述的异常是否达到阈值”。

因此，本节结果可以说明当前 Caption-E0 管线的整体性能低于 Video-E0，但不能把全部差异严格归因于视频输入被替换成 caption。若论文需要严格的“仅输入改变”结论，应使用同一个 VideoLLaMA3 checkpoint 和匹配的阈值 prompt，对视频输入与 caption 文本输入进行重新对比。

### Caption-E0 分数统计

| 数据集 | clip 数 | 唯一值数量 | 均值 ± 标准差 | 最大重复值占比 |
|---|---:|---:|---:|---:|
| UCF-Crime | 69,634 | 48,540 | 0.188053 ± 0.203280 | 1.0555% |
| MSAD（240 视频评测子集） | 9,250 | 7,147 | 0.263882 ± 0.279569 | 0.2919% |
| XD-Violence | 146,449 | 128,668 | 0.312178 ± 0.286589 | 0.1332% |

Caption-E0 本身也是连续 likelihood 分数，并不存在 Generation 的 11 值量化问题；它与 Video-E0 的差异主要来自视觉信息被 caption 压缩后的信息损失。

## 3. 当前结论

1. **Likelihood 明显优于直接生成分数。** 直接生成虽然记录的格式解析失败率为 0，但分数几乎被压缩到 11 个离散值，产生大量 ties，限制了帧级排序性能。
2. **直接访问原始视频总体优于 Caption-E0。** 三个数据集上 Video-E0 的 ROC-AUC、PR-AUC 和 Max-F1 均高于对应 Caption-E0，UCF 和 MSAD 的差距尤其明显。
3. **时间顺序消融结果：** 正常顺序的 ROC-AUC 只比 Shuffle 高约 0.13 个百分点，但 PR-AUC 和 Max-F1 分别高约 0.98 和 0.86 个百分点；Single-frame 明显下降，说明多帧输入包含有用的动态/上下文信息，但模型的 ROC 排序对帧顺序本身并不极端敏感。

## 4. 时间顺序与单帧消融（UCF-Crime）

该实验保持 E0 的视频窗口、采样帧数、阈值 likelihood、PAVA、插值和高斯平滑不变，只改变输入帧序列。Shuffle 使用固定 seed=17；Single-frame 使用窗口中心帧。

| 方法 | ROC-AUC | PR-AUC | Max-F1 | 相对 E0-Order 的 ROC / PR / F1 |
|---|---:|---:|---:|---:|
| E0-Order（复用已有结果） | **86.2693** | **39.4706** | **45.2153** | — |
| E0-Shuffle | 86.1425 | 38.4937 | 44.3540 | −0.1268 / −0.9769 / −0.8613 |
| E0-Single-frame | 83.2048 | 28.8855 | 36.7180 | −3.0645 / −10.5852 / −8.4973 |

Caption-E0 rerun 得到 80.2765% / 32.5501% / 39.0150%，与之前的 Caption-E0 结果完全一致，说明 UCF Caption-E0 的复现实验没有执行或评测问题。

## 结果文件位置

- UCF Generation：`data/ucf_crime/scores/videollama3_direct/metrics_final_eval/`
- MSAD Generation：`data/MSAD/scores/videollama3_direct_generated_stride16/metrics/`
- XD Generation：`data/xd_violence/scores/videollama3_direct_generated_stride16/metrics_final_eval/`
- UCF Caption-E0：`data/ucf_crime/scores/caption_e0_likelihood_stride16/metrics_final_eval/`
- MSAD Caption-E0：`data/MSAD/scores/caption_e0_likelihood_stride16/metrics_final_anomaly_test240/`
- XD Caption-E0：`data/xd_violence/scores/caption_e0_likelihood_stride16/metrics_final_eval/`
