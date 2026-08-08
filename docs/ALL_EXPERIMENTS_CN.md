# COVAS-VAD 全部实验归档

> 快照日期：2026-08-05  
> 项目：COVAS-VAD（Cumulative Ordinal Visual Anomaly Scoring for Video Anomaly Detection）  
> 本文档汇总当前仓库中与 COVAS-VAD/URF-HVAA 研究直接相关的主实验、作者对比、模型消融、输入消融、后处理、历史 refinement 和计算开销实验。没有独立完整指标的实验明确标为 **pilot**、**partial** 或 **diagnostic**。

## 0. 状态和指标约定

- **complete**：指定测试集完整完成，score JSON 和指标均存在。
- **partial/pilot**：只完成部分视频或小规模试验。
- **diagnostic**：只进行离线分析或后处理，不重新调用 VideoLLaMA3。
- **running**：已经产生部分结果但尚未结束。
- **not promoted**：有结果或代码，但没有被选为主方法。

除特别说明外，指标顺序均为 ROC-AUC / PR-AUC / Max-F1，单位为百分比。作者方法数字来自 URF-HVAA/VADTree 公开结果，不是本仓库重新实现的作者模型。

## 1. 数据集和统一设置

| 数据集 | 测试视频 | 完整 clip 数 | 正常标签 | 主结果目录 |
|---|---:|---:|---:|---|
| UCF-Crime | 290 | 69,634 | 7 | results/ucf_crime/ |
| MSAD | 240 | 9,250 | 0 | results/msad/ |
| XD-Violence | 800 | 146,449 | 4 | results/xd_violence/ |

E0 默认参数：

| 参数 | 设置 |
|---|---:|
| 模型 | VideoLLaMA3-7B |
| 窗口长度 | 10 秒 |
| 中心帧间隔 | 16 原视频帧 |
| 采样 FPS | 2 |
| 最大输入帧数 | 10 |
| 阈值 | 0.1, 0.2, ..., 1.0 |
| 候选 | 单 token YES/NO |
| likelihood temperature | 1.0 |
| PAVA | 开启，非递增投影 |
| frame-level 平滑 | Gaussian，sigma=10 |
| caption/标签/refinement/fusion | 不使用 |

输出 JSON 与原 URF-HVAA 兼容，键为中心帧编号：

~~~json
{"0": 0.1032, "16": 0.1176, "32": 0.6814}
~~~

## 2. 主方法 E0：原始视频累计阈值 likelihood

### 2.1 流程

~~~text
原始视频
  → 10 秒滑动窗口
  → 均匀采样最多 10 帧
  → VideoLLaMA3 直接观察原始视频
  → 10 个有序累计阈值
  → 读取 YES/NO logits
  → 10 个尾概率
  → PAVA 单调投影
  → 尾概率均值得到连续 clip 分数
  → 平滑并展开到 frame-level
  → ROC-AUC / PR-AUC / Max-F1
~~~

对阈值 $$\tau_k=0.1k$$：

$$
p_k=P(S\geq \tau_k\mid V)
=\operatorname{softmax}(\ell_{NO},\ell_{YES})_{YES}.
$$

PAVA 后的连续分数为：

$$
s(V)=\frac{1}{10}\sum_{k=1}^{10}\tilde p_k.
$$

十个阈值共享一次视频解码和视觉编码，文本后缀按 batch 计算。

### 2.2 三数据集完整指标

| 数据集 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| UCF-Crime | **86.2693** | 39.4706 | **45.2153** |
| MSAD | **94.3630** | **81.2026** | **76.2655** |
| XD-Violence | **92.1099** | **75.3590** | **73.1337** |
| 三数据集平均 | **90.9141** | **65.3441** | 64.8715 |

### 2.3 与作者结果对比

| 数据集 | 方法 | ROC-AUC | PR-AUC | Max-F1 |
|---|---|---:|---:|---:|
| UCF-Crime | VADTree（作者） | 84.74 | **41.89** | 44.29 |
| UCF-Crime | URF-HVAA（作者） | 84.36 | 36.15 | 41.83 |
| UCF-Crime | **COVAS-VAD E0** | **86.27** | 39.47 | **45.22** |
| MSAD | VADTree（作者） | 89.32 | 71.41 | 68.80 |
| MSAD | URF-HVAA（作者） | 93.06 | 77.81 | 74.82 |
| MSAD | **COVAS-VAD E0** | **94.36** | **81.20** | **76.27** |
| XD-Violence | VADTree（作者） | 90.47 | 67.91 | 69.17 |
| XD-Violence | URF-HVAA（作者） | 91.34 | 68.07 | 71.93 |
| XD-Violence | **COVAS-VAD E0** | **92.11** | **75.36** | **73.13** |

UCF-Crime 的 PR-AUC 仍低于 VADTree，因此论文应写“总体提升”或“多数指标提升”，不要写成所有指标均最优。

## 3. 核心对比：Generation、11 类 likelihood、E0

三者均直接输入原始视频：

- **Generation**：模型生成 [0.0] 至 [1.0] 的一个离散分数；
- **11-class likelihood**：对 0.0–1.0 的 11 个候选答案计算条件概率并求期望；
- **E0**：十个有序累计阈值的尾概率均值。

| 数据集 | Generation | 11-class likelihood | E0 |
|---|---:|---:|---:|
| UCF-Crime | 70.2006 / 25.7273 / 30.5620 | 84.4313 / 40.2548 / 44.7530 | **86.2693 / 39.4706 / 45.2153** |
| MSAD | 85.7152 / 67.2532 / 60.9461 | 92.4355 / 79.1942 / 75.1304 | **94.3630 / 81.2026 / 76.2655** |
| XD-Violence | 86.1817 / 68.7754 / 71.2585 | 91.7651 / 77.1522 / 74.6449 | **92.1099 / 75.3590 / 73.1337** |

Generation 三个数据集解析失败率均为 0，但分数只有约 9–11 个离散值；likelihood 保留了几乎每个 clip 的连续排序信息。

结果目录：

~~~text
results/experiments/<dataset>/direct_generation/
results/experiments/<dataset>/direct_11class_likelihood_promptfix/
~~~

## 4. Caption 信息瓶颈实验

### 4.1 Video-E0 vs. Caption-E0

~~~text
Video-E0：原始视频 clip → VideoLLaMA3 → 累计阈值 likelihood
Caption-E0：原始视频 clip → caption → Llama 3.1 → 累计阈值 likelihood
~~~

| 数据集 | Video-E0 | Caption-E0 |
|---|---:|---:|
| UCF-Crime | **86.2693 / 39.4706 / 45.2153** | 80.2765 / 32.5501 / 39.0150 |
| MSAD | **94.3630 / 81.2026 / 76.2655** | 92.1396 / 78.8225 / 75.2398 |
| XD-Violence | **92.1099 / 75.3590 / 73.1337** | 91.9873 / 74.6177 / 71.7832 |

结果总体支持直接访问原始视频优于 caption 中间表示，UCF 和 MSAD 的差距最明显。

严格解释限制：Caption-E0 使用 Llama 3.1 文本模型，prompt 也不是 VideoLLaMA3 prompt 的完全同构版本，因此这是完整管线对照，不是只改变输入模态的严格单变量实验。需要严格因果结论时，应使用同一 checkpoint 和匹配 prompt 重跑。

### 4.2 Controlled Visual vs. Caption Input

该实验使用同一个 VideoLLaMA3 checkpoint、相同的 system prompt、相同的十个阈值、YES/NO 候选、temperature、PAVA、积分和平滑流程；唯一变化是输入为原始视频或对应 clip caption。已有 VideoLLaMA3 caption 被直接复用，不重新生成 caption。

| 数据集 | Visual-E0 | Controlled Caption-E0 |
|---|---:|---:|
| UCF-Crime | **86.2693 / 39.4706 / 45.2153** | 80.7024 / 33.1725 / 39.8348 |
| MSAD | **94.3630 / 81.2026 / 76.2655** | 93.9672 / 80.8616 / 74.9790 |
| XD-Violence | **92.1099 / 75.3590 / 73.1337** | 91.7000 / 69.6727 / 71.8756 |

完整性：UCF 290/290、MSAD 240/240、XD 800/800，三者均无 score error。相对 Visual-E0，Controlled Caption-E0 的 ROC-AUC / PR-AUC / Max-F1 变化分别为：UCF −5.5669 / −6.2981 / −5.3805，MSAD −0.3958 / −0.3410 / −1.2865，XD −0.4098 / −5.6863 / −1.2581 个百分点。

该结果是当前最严格的输入条件对照，说明即便使用相同模型和匹配评分 prompt，直接视觉输入仍整体优于 caption 输入。旧 Caption-E0 结果仍保留用于历史比较，但不能与本节的 controlled 结果混写。

结果目录：

~~~text
data/ucf_crime/scores/caption_e0_controlled_videollama3_stride16/
data/MSAD/scores/caption_e0_controlled_videollama3_stride16/
data/xd_violence/scores/caption_e0_controlled_videollama3_stride16/
~~~

### 4.3 原始 URF-HVAA Caption/Refinement 管线

历史流程：

~~~text
VideoLLaMA3 caption
  → Llama3.1 初始分数
  → score_filter 选择区间
  → summarize_window 提取异常标签
  → refine_with_tag 二次评分
~~~

它被保留用于 baseline 和成本研究，但不属于 COVAS-VAD 主方法。主方法不生成 caption、异常标签，也不执行 refinement。

## 5. E1/E2/E3 选项校准消融（MSAD）

### E1：单向 A/B

~~~text
A: The anomaly severity is at least τ.
B: The anomaly severity is strictly below τ.
~~~

$$
m=\ell_A-\ell_B,\qquad p=\sigma(m/T).
$$

| 方法 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| E0 | **94.3630** | **81.2026** | **76.2655** |
| E1 | 93.2254 | 78.9846 | 75.6784 |

### E2：A/B 正反语义交换

$$
m_1=\ell_A^{forward}-\ell_B^{forward},
$$

$$
m_2=\ell_B^{swap}-\ell_A^{swap},
$$

$$
p=\sigma\left(\frac{m_1+m_2}{2T}\right).
$$

| 方法 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| E2 | 93.4871 | 79.4267 | 75.7109 |

E2 高于 E1，说明交换抵消了一部分 A/B 位置偏置，但仍低于 E0，且问题数约翻倍。

### E3：YES/NO 极性反转

正向问题是“severity at least τ”，反向问题是“severity strictly below τ”：

$$
m_1=\ell_{YES}^{forward}-\ell_{NO}^{forward},
$$

$$
m_2=\ell_{NO}^{reverse}-\ell_{YES}^{reverse},
$$

$$
p=\sigma\left(\frac{m_1+m_2}{2T}\right).
$$

| 方法 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| E3 | 93.9734 | 80.3436 | 76.1857 |

E3 接近 E0，但没有超过 E0，且文本侧问题数翻倍，因此不作为主方法。

结果目录：

~~~text
results/experiments/MSAD/e1_ab/
results/experiments/MSAD/e2_ab_swap/
results/experiments/MSAD/e3_yesno_polarity_swap/
~~~

## 6. 时间步长、阈值数和帧输入消融

### 6.1 Stride16 vs. Stride32 + 插值

Stride32 只将模型评分中心从 16 帧改为 32 帧，完成后线性插值回 16 帧；窗口、FPS、帧数、prompt、阈值和评测平滑不变。

| 数据集 | E0 Stride16 | Stride32 + 插值 |
|---|---:|---:|
| UCF-Crime | 86.2693 / 39.4706 / 45.2153 | 86.1638 / 39.3812 / 45.2508 |
| MSAD | 94.3630 / 81.2026 / 76.2655 | 94.1911 / 80.6586 / 76.5541 |
| XD-Violence | 92.1099 / 75.3590 / 73.1337 | 92.0858 / 75.2792 / 73.0557 |

Stride32 约减少一半评分 clip，性能变化很小，是有效的成本消融，但 E0 Stride16 仍是主结果。

### 6.2 10 阈值 vs. 5 阈值

5 阈值为 0.2、0.4、0.6、0.8、1.0，只改变阈值聚合。

| 数据集 | Stride32-10T | Stride32-5T |
|---|---:|---:|
| UCF-Crime | 86.1638 / 39.3812 / 45.2508 | 86.2315 / 39.4574 / 45.6888 |
| MSAD | 94.1911 / 80.6586 / 76.5541 | 94.1871 / 80.5262 / 76.4265 |
| XD-Violence | 92.0858 / 75.2792 / 73.0557 | 92.0692 / 75.1224 / 72.9330 |

5 阈值减少阈值判断，但视觉编码不变，因此总耗时不会减半，且没有稳定超过 10 阈值 E0。

### 6.2.1 Stride16 五阈值实际推理结果

随后在三个数据集上完成了真实的 Stride16 五阈值推理，使用同一套评测接口：

| 数据集 | E0-10T Stride16 | E0-5T Stride16 |
|---|---:|---:|
| UCF-Crime | 86.2693 / 39.4706 / 45.2153 | 86.3231 / 39.5126 / **45.6158** |
| MSAD | **94.3630 / 81.2026 / 76.2655** | 94.3194 / 80.9742 / 76.2539 |
| XD-Violence | **92.1099 / 75.3590 / 73.1337** | 92.0924 / 75.1918 / 72.9999 |

Stride16 五阈值在 UCF 的 Max-F1 略高，但在 MSAD 和 XD 的 PR-AUC/ROC-AUC 略低；因此它是有价值的阈值数消融，不替代统一主方法 E0-10T。

实际结果目录：

~~~text
data/ucf_crime/scores/videollama3_cumulative_likelihood_threshold5_actual_stride16/
data/MSAD/scores/videollama3_cumulative_likelihood_threshold5_actual_stride16/
data/xd_violence/scores/videollama3_cumulative_likelihood_threshold5_actual_stride16/
~~~

### 6.3 Stride32 + 8 帧 pilot

在 Stride32 上将每个 clip 的均匀 RGB 输入从 10 帧减为 8 帧。UCF 仅完成 100 视频 pilot，无完整三数据集指标，不用于主表。

结果：results/experiments/ucf_crime/e0_stride32_f8_pilot100/。

### 6.4 正常顺序、Shuffle、Single-frame（UCF）

| 方法 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| E0-Order | **86.2693** | **39.4706** | **45.2153** |
| E0-Shuffle（seed=17） | 86.1425 | 38.4937 | 44.3540 |
| E0-Single-frame | 83.2048 | 28.8855 | 36.7180 |

多帧上下文明显重要；Shuffle 只造成小幅下降，说明排序同时依赖静态外观和时间证据；单帧会明显损失上下文。

### 6.5 E4/S1/S2 中心密集采样（MSAD）

只改变 10 秒窗口内 10 帧的位置：4 帧覆盖全局，6 帧密集覆盖中心。

| 方法 | 采样方式 | ROC-AUC | PR-AUC | Max-F1 |
|---|---|---:|---:|---:|
| E0 | 10 帧均匀 | 94.3630 | 81.2026 | 76.2655 |
| S1 / E4-S4 | 4 帧全局 + 6 帧中心 4 秒 | 94.3493 | **81.2101** | 76.2402 |
| S2 / E4-S2 | 4 帧全局 + 6 帧中心 2 秒 | 94.2867 | 81.0542 | 76.1688 |

中心密集采样没有稳定提升；S1 基本持平，S2 略降。

## 7. 离线后处理、不确定性和选择性核查

这些实验不重新调用 VideoLLaMA3，只使用 E0 已保存的尾概率或 clip 分数。

### 7.1 跨窗口投影/Fusion

| 数据集 | 方法 | ROC-AUC | PR-AUC | Max-F1 |
|---|---|---:|---:|---:|
| UCF-Crime | Adjacent mean | 86.2681 | 39.4580 | 45.2041 |
| UCF-Crime | Overlap mean | 86.2559 | 39.2315 | 45.0517 |
| MSAD | Adjacent mean | 94.3633 | 81.1971 | 76.2539 |
| MSAD | Overlap mean | 94.3081 | 80.9324 | 76.3035 |
| XD-Violence | Adjacent mean | 92.1068 | 75.3417 | 73.1281 |
| XD-Violence | Overlap mean | 92.0797 | 75.1972 | 73.0561 |

另有 adaptive、overlap_logit 等结果。它们没有稳定超过 E0，且增加人工后处理先验，因此不作为主方法。目录：results/experiments/<dataset>/temporal_fusion/。

### 7.2 PAVA、尾概率熵和 residual entropy

已分析十个尾概率是否分散、PAVA 修改量 $$D=|p-	ilde p|$$、尾概率熵是否包含分数之外的新信息，以及 E0 预测正常 clip 中 FN/TN 的熵、修改量和 residual entropy。该阶段用于诊断和选择性核查设计，不使用测试标签选择正式模型参数，未纳入主表。

### 7.3 单阈值、加权和自适应序数聚合

保留了 p0.3、p0.5、p0.7 单阈值、十阈值等权 E0、按严重度中心的高斯加权、单调 sigmoid 序数曲线拟合等离线探索。主方法保持等权 E0，避免直接使用三个测试集搜索权重或阈值。

### 7.4 近边界中心核查

e0_selective_center_verify 只选择每个视频内约 85%–95% 分位的候选 clip，重新用中心 4 秒均匀采样 10 帧并执行 E0 评分，保存：

$$
Delta s=s_{verify}-s_{E0}.
$$

该实验用于 FN/TN 分析，尚未形成三数据集完整主指标。

### 7.5 Gaussian smoothing sigma 消融

直接读取 E0 clip 分数，只改变一维 Gaussian smoothing 的 sigma；frame interval、插值/重复方式、标签和评测接口完全不变。sigma=0 表示不平滑。

| 数据集 | sigma | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|---:|
| UCF-Crime | 0 | 84.8153 | 37.6369 | 43.4400 |
| UCF-Crime | 2 | 85.7311 | 39.8825 | 45.3063 |
| UCF-Crime | 5 | 86.1208 | **40.6049** | **45.6038** |
| UCF-Crime | 10 | **86.2693** | 39.4706 | 45.2153 |
| UCF-Crime | 20 | 85.9467 | 36.6442 | 43.9617 |
| MSAD | 0 | 94.2967 | 81.9810 | 77.1918 |
| MSAD | 2 | 94.5418 | **82.6604** | **77.4001** |
| MSAD | 5 | **94.5918** | 82.4643 | 76.7605 |
| MSAD | 10 | 94.3630 | 81.2026 | 76.2655 |
| MSAD | 20 | 93.9017 | 79.1658 | 75.6188 |
| XD-Violence | 0 | 91.4096 | 74.5722 | 72.3337 |
| XD-Violence | 2 | 92.0382 | 76.1349 | 73.4700 |
| XD-Violence | 5 | **92.2261** | **76.4149** | **73.5152** |
| XD-Violence | 10 | 92.1099 | 75.3590 | 73.1337 |
| XD-Violence | 20 | 91.6475 | 73.2055 | 71.8817 |

结果显示过强平滑（sigma=20）会损害性能；sigma=2–5 在 PR-AUC/Max-F1 上通常更好，而 sigma=10 的 ROC-AUC 在 UCF 仍最高。由于这些数值来自测试集，不能直接据此修改主方法，正式配置仍保持 sigma=10，除非后续在验证集确定统一 sigma。

指标文件位置：各 E0 score 目录下的 `metrics_sigma_sweep/sigma_<value>/`。

## 8. E5-SDEE 动态证据 pilot

E5-SDEE 将 10 张静态 RGB 替换为交替的状态帧和运动证据图：

~~~text
10 秒窗口
  → 5 个 2 秒区间
  → S1, M1, S2, M2, ..., S5, M5
  → E0 的 YES/NO 十阈值评分
~~~

运动图第一版使用全局运动补偿后的帧间变化（D1），不改变模型、阈值、PAVA 和评测。当前只完成 UCF 小规模 order/shuffle/no-align pilot，无三数据集完整指标。

结果目录：results/experiments/ucf_crime/e5_sdee_pilot_20/。

## 9. 历史 Caption、标签和 Refinement 探索

以下目录属于历史探索，不是主方法：

~~~text
results/experiments/ucf_crime/legacy_refinement/direct_likelihood_tag_refinement/
results/experiments/ucf_crime/legacy_refinement/direct_likelihood_gate_049_051/
results/experiments/ucf_crime/legacy_refinement/direct_generation_tag_refinement/
~~~

UCF 还保存了 caption sampling、category generalization、Modify2 full-290、multi-scale caption fusion、VLM-only dense 2s、zero-shot anticipation、zero-shot hazard/survival gate 等 exploratory 输出。没有 metrics 或标为 partial 的目录只能作为 pilot/qualitative 证据。

## 10. 计算开销实验

src/cost_benchmark_50.py 固定每个数据集前 50 个官方测试视频，依次执行：

1. 原始视频 E0-10；
2. 原始视频 E0-5；
3. 原始 Caption → Llama 分数 → score_filter → 标签提取 → refinement。

### 10.1 E0-10 vs. E0-5

| 数据集 | 版本 | clips | 总时间 | 秒/clip | 阈值判断数 | 相对 E0-10 |
|---|---|---:|---:|---:|---:|---:|
| UCF-Crime | E0-10 | 13,688 | 1:11:40 | 0.314 | 136,880 | — |
| UCF-Crime | E0-5 | 13,688 | 1:02:05 | 0.272 | 68,440 | **快 13.4%** |
| MSAD | E0-10 | 1,681 | 1:58:37 | 4.233 | 16,810 | — |
| MSAD | E0-5 | 1,681 | 1:50:47 | 3.954 | 8,405 | **快 6.6%** |
| XD-Violence | E0-10 | 8,424 | 1:20:18 | 0.572 | 84,240 | — |
| XD-Violence | E0-5 | 8,424 | 1:13:30 | 0.524 | 42,120 | **快 8.5%** |

减少阈值只减少文本判断，视觉编码不变，因此总耗时不会减半。成本 score JSON 位于：

~~~text
data/cost_benchmark_50/<dataset>/E0-10/
data/cost_benchmark_50/<dataset>/E0-5/
~~~

### 10.2 Caption+Refine 成本阶段

该阶段分别记录 caption generation、caption scoring、tag extraction 和 refinement 的视频/clip 数、耗时、吞吐量、显存和输出大小。当前快照中该成本任务尚未形成完整测试集指标，应以：

~~~text
data/cost_benchmark_50/benchmark_config.json
src/cost_benchmark_50.py
~~~

为准。它是 50-video 成本 benchmark，不应和 290/240/800 视频完整指标混用。

## 11. 代码、结果和复现入口

### 11.1 主方法和工具

~~~text
src/video_cumulative_score.py
src/video_score_utils.py
src/eval.py
scripts/run_video_cumulative_rebalanced.sh
scripts/run_video_cumulative_stride32_rebalanced.sh
scripts/run_threshold5_3dataset.sh
~~~

### 11.2 主要消融代码

~~~text
src/video_direct_score.py
src/video_cumulative_stride32.py
src/video_cumulative_threshold5.py
src/video_ab_calibrated_score.py
src/video_yesno_calibrated_score.py
src/video_center_dense_score.py
src/video_temporal_order_ablation.py
src/cost_benchmark_50.py
~~~

### 11.3 Caption baseline 代码

~~~text
src/video_pre_caption.py
src/llm_anomaly_scorer.py
src/score_filter.py
src/summarize_window.py
src/refine_with_tag.py
src/video_refine_with_tag.py
~~~

这些文件只用于 baseline/审计，不是 COVAS-VAD E0 的必要步骤。

### 11.4 结果索引

~~~text
COVAS-VAD/results/EXPERIMENT_MANIFEST.json
COVAS-VAD/docs/EXPERIMENTS_CN.md
COVAS-VAD/docs/ABLATION_OFFLINE_RESULTS_CN.md
COVAS-VAD/docs/CURRENT_BEST_METHOD_CN_URF_HVAA.md
~~~

manifest 区分 complete、partial 和探索性目录，论文主表只使用完整结果。

## 12. 最终结论

当前应作为论文主方法的版本是：

> **COVAS-VAD E0：原始视频直接输入 + YES/NO 累计阈值 likelihood + PAVA + 等权尾概率积分。**

它：

1. 不生成 caption；
2. 不生成异常标签；
3. 不进行二次 refinement；
4. 不使用 fused score 或数据集特定人工权重；
5. 直接保留原始视频视觉证据；
6. 显式利用异常严重度的序数关系；
7. 输出连续分数并减少离散 ties；
8. 支持共享视觉编码、批量阈值推理、断点续跑和多 GPU。

现有实验支持以下结论：

- likelihood 明显优于自由生成的离散分数；
- 直接视频条件总体优于 Caption 条件；
- E0 的自然 YES/NO 语义优于 A/B 校准版本；
- 10 阈值比 5 阈值更稳健，但 5 阈值可作为成本消融；
- Stride32 可大幅降低计算量，性能变化较小；
- 多帧输入优于单帧，Shuffle 只造成小幅下降；
- 中心密集采样、跨窗口 fusion、标签 refinement 和人工加权均未稳定超过 E0；
- 因此当前最有解释性、最符合 training-free 定位的版本仍是 E0。

## 13. 不应列入主结果的内容

以下内容保留用于历史追踪，但不属于 COVAS-VAD 主方法：

- URF-HVAA 的 caption/tag/refinement；
- direct/refined/fused 混合分数；
- 根据测试集标签搜索 Gate、权重或阈值；
- G3、HGTree、作者方法重实现等无关探索；
- 只有少量视频或没有完整指标的 pilot；
- 只统计成本但没有完整测试集指标的 50-video benchmark。

## 14. 结果目录完整索引

下面列出 manifest 中已经归档的实验目录。目录存在不代表都有完整指标；是否可以写入论文主表，以 `COVAS-VAD/results/EXPERIMENT_MANIFEST.json` 的状态为准。

### UCF-Crime

| 状态 | 实验目录 | 说明 |
|---|---|---|
| complete | `caption_e0_likelihood_stride16` | Caption-E0 |
| complete | `direct_generation` | 直接生成分数 |
| complete | `direct_11class_likelihood_promptfix` | 11 类候选 likelihood |
| complete | `direct_likelihood_legacy` | 早期直接 likelihood，历史记录 |
| complete | `e0_stride32` | Stride32 + 插值 |
| complete | `e0_stride32_threshold5` | Stride32 + 五阈值离线聚合 |
| complete | `e0_shuffle` | 顺序打乱 |
| complete | `e0_single_frame` | 中心单帧 |
| partial | `e0_stride32_f8_pilot100` | 8 帧 pilot |
| diagnostic | `e0_selective_center_verify` | 近边界中心 4 秒核查 |
| pilot | `e5_sdee_pilot_20` | SDEE order/shuffle/no-align |
| diagnostic | `temporal_fusion` | 跨窗口后处理 |
| historical | `legacy_refinement` | 标签/refinement 历史实验 |
| exploratory | `exploratory` | caption sampling、类别泛化、Modify2、dense 2s、anticipation、hazard/survival gate |
| partial | `direct_likelihood_stride32` | 断点目录 |
| partial | `direct_likelihood_stride32_kvcache` | KV-cache 断点目录 |

### MSAD

| 状态 | 实验目录 | 说明 |
|---|---|---|
| complete | `caption_e0_likelihood_stride16` | Caption-E0 |
| complete | `direct_generation` | 直接生成分数 |
| complete | `direct_11class_likelihood_promptfix` | 11 类候选 likelihood |
| complete | `direct_11class_likelihood_legacy` | 早期直接 likelihood，历史记录 |
| complete | `e0_stride32` | Stride32 + 插值 |
| complete | `e0_stride32_threshold5` | Stride32 + 五阈值离线聚合 |
| complete | `e1_ab` | 单向 A/B |
| complete | `e2_ab_swap` | A/B 交换校准 |
| complete | `e3_yesno_polarity_swap` | YES/NO 极性反转 |
| complete | `e4_center_dense_s2` | 中心 2 秒密集采样 |
| complete | `e4_center_dense_s4` | 中心 4 秒密集采样 |
| diagnostic | `temporal_fusion` | 跨窗口后处理 |
| partial | `direct_likelihood_stride32` | 断点目录 |
| partial | `direct_likelihood_stride32_kvcache` | KV-cache 断点目录 |

### XD-Violence

| 状态 | 实验目录 | 说明 |
|---|---|---|
| complete | `caption_e0_likelihood_stride16` | Caption-E0 |
| complete | `direct_generation` | 直接生成分数 |
| complete | `direct_11class_likelihood_promptfix` | 11 类候选 likelihood |
| complete | `direct_11class_likelihood_legacy` | 早期直接 likelihood，历史记录 |
| complete | `e0_stride32` | Stride32 + 插值 |
| complete | `e0_stride32_threshold5` | Stride32 + 五阈值离线聚合 |
| diagnostic | `temporal_fusion` | 跨窗口后处理 |
| partial | `direct_likelihood_stride32` | 断点目录 |
| partial | `direct_likelihood_stride32_kvcache` | KV-cache 断点目录 |

### 成本 benchmark

成本实验不在 `COVAS-VAD/results/experiments` 中，而在：

~~~text
data/cost_benchmark_50/
├── ucf_crime/{E0-10,E0-5,caption_text,caption_scores,caption_refined_raw}/
├── MSAD/{E0-10,E0-5,caption_text,caption_scores,caption_refined_raw}/
└── xd_violence/{E0-10,E0-5,caption_text,caption_scores,caption_refined_raw}/
~~~

E0-10、E0-5 的 score JSON 已保存；Caption+Refine 只用于成本统计，不能直接当作三个完整测试集的性能结果。
