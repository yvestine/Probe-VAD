# 原始视频直接异常评分：E0–E4、S1–S2 实验汇总

> 更新时间：2026-07-31  
> 当前推荐方法：**E0——YES/NO 累计阈值似然评分**  
> 本文只汇总 E0–E4 及 S1、S2，不包含任何跨窗口 Fusion 结果。

## 1. 总体结论

当前可以正式确定的最佳版本是 **E0**。它让 VideoLLaMA3 直接观察原始
视频窗口，通过 10 个有序 YES/NO 阈值问题得到连续异常分数，不生成
caption、异常标签，不执行 refinement，也不融合其他方法的分数。

MSAD 完整消融的主要结论如下：

1. E0 的 ROC-AUC 和 Max-F1 最高，且计算量低于 E2、E3。
2. E1 明显低于 E0，说明具有自然语义的 YES/NO 比中性的 A/B 更适合当前模型。
3. E2 高于 E1，说明交换 A/B 语义可以抵消部分选项偏置，但仍无法恢复 E0 的性能。
4. E3 接近 E0，但没有超过 E0，说明 YES/NO 固定偏置不是当前主要瓶颈。
5. E4-S1 与 E0 基本持平；中心密集采样没有稳定提升排序性能。
6. E4-S2 比 S1 略差，说明把 6 帧进一步压缩到中心 2 秒会损失必要的时间上下文。

## 2. 版本命名

| 名称 | 改动 | 候选答案 | 每个 clip 的问题数 |
|---|---|---|---:|
| E0 | 原始累计阈值似然 | YES/NO | 10 |
| E1 | 将 YES/NO 换成单向 A/B | A/B | 10 |
| E2 | 正向与交换语义的 A/B margin 校准 | A/B | 20 |
| E3 | 正向与互补问题的 YES/NO margin 校准 | YES/NO | 20 |
| E4 | 保持 E0 评分，只改变视频帧采样位置 | YES/NO | 10 |
| S1 | E4 配置：4 帧全局 + 6 帧覆盖中心 4 秒 | YES/NO | 10 |
| S2 | E4 配置：4 帧全局 + 6 帧覆盖中心 2 秒 | YES/NO | 10 |

S1、S2 是 E4 的两个采样配置，不是新的语言评分接口。底层结果目录仍沿用
`e4_center_dense_g4_c6_s4` 和 `e4_center_dense_g4_c6_s2`。

## 3. 所有版本共享的基础流程

```text
原始视频
  → 10 秒滑动窗口
  → VideoLLaMA3 直接观察当前视频窗口
  → 10 个有序阈值的候选答案 logits
  → 10 个异常尾概率
  → PAVA 单调递减投影
  → 尾概率均值得到 clip 异常分数
  → clip-level Gaussian 平滑（sigma=10）
  → 按 16 帧间隔展开到 frame-level
  → ROC-AUC、PR-AUC、Max-F1
```

整个 E0–E4 流程均不包含：

- 视频 caption；
- caption 输入 Llama3.1；
- 可疑区间异常标签；
- 标签条件 refinement；
- initial/refined 加权；
- URF-HVAA 或 VADTree 分数融合；
- 跨窗口 score fusion。

### 3.1 视频窗口和采样参数

| 参数 | 设置 |
|---|---:|
| 窗口长度 | 10 秒 |
| 相邻中心帧间隔 | 16 个原视频帧 |
| 候选采样率 | 2 FPS |
| 输入模型的最大帧数 | 10 帧 |
| 阈值 | 0.1, 0.2, ..., 1.0 |
| likelihood temperature | 1.0 |
| 阈值 batch size | 10 |
| 单调投影 | PAVA |
| 评测平滑 | clip-level Gaussian, sigma=10 |

输出 JSON 的键是原视频中心帧号，值是当前 clip 的连续异常分数：

```json
{
  "0": 0.083412,
  "16": 0.091527,
  "32": 0.274188
}
```

### 3.2 共同异常检测指令

```text
You are a careful video anomaly detector for public surveillance scenes.
Judge only visible evidence in the current video segment. Use an ordered
anomaly-severity scale from 0 to 1: 0 means ordinary safe activity; 0.5
means clearly concerning or plausibly anomalous activity; and 1 means
unmistakably severe, dangerous, violent, criminal, or emergency activity.
Intermediate thresholds preserve this order. Do not infer events that are
not visually supported.
```

### 3.3 PAVA 和最终连续分数

对 10 个阈值分别估计尾概率：

$$
p_k=P(S\ge \tau_k\mid V),\qquad
\tau_k\in\{0.1,0.2,\ldots,1.0\}.
$$

理论上尾概率应满足：

$$
p_1\ge p_2\ge\cdots\ge p_{10}.
$$

使用 PAVA 将其投影为单调递减序列
\(\tilde p_1,\ldots,\tilde p_{10}\)，最终分数为：

$$
s(V)=\frac{1}{10}\sum_{k=1}^{10}\tilde p_k.
$$

它具有异常等级期望的解释，并且不会把结果限制为 11 个离散值。

## 4. E0：YES/NO 累计阈值似然

### 做法

每个阈值只询问一次：

```text
On the severity scale defined by the system, is the anomaly severity
visible in this video segment at least {threshold:.1f}?
Judge only the current video's visual evidence. Answer exactly YES or NO.
```

读取候选位置的 YES、NO logits：

$$
m_k=\ell_{\mathrm{YES}}-\ell_{\mathrm{NO}},
\qquad
p_k=\sigma(m_k/T).
$$

10 个阈值共享同一次视频解码和视觉编码，并批量完成文本侧阈值判断。

### 效果

E0 是当前唯一在 UCF-Crime、MSAD 和 XD-Violence 三个完整测试集上统一完成的
核心版本：

| 数据集 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| UCF-Crime | **86.2693%** | **39.4706%** | **45.2153%** |
| MSAD | **94.3630%** | **81.2026%** | **76.2655%** |
| XD-Violence | **92.1099%** | **75.3590%** | **73.1337%** |
| 三数据集算术平均 | **90.9141%** | **65.3441%** | **64.8715%** |

数据完整性：

| 数据集 | 完整视频 | 完整 clips |
|---|---:|---:|
| UCF-Crime | 290/290 | 69,634/69,634 |
| MSAD | 240/240 | 9,250/9,250 |
| XD-Violence | 800/800 | 146,449/146,449 |

## 5. E1：单向 A/B 阈值判断

### 做法

将 E0 的 YES/NO 问题改成固定顺序的两个选项：

```text
A: The anomaly severity is at least τ.
B: The anomaly severity is strictly below τ.
```

计算：

$$
m_k=\ell_A-\ell_B,\qquad p_k=\sigma(m_k/T).
$$

E1 用来检验中性 A/B token 是否比自然语言 YES/NO 更适合模型。

### 效果

| 数据集 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| MSAD | 93.2254% | 78.9846% | 75.6784% |
| 相对 E0 | −1.1377 | −2.2181 | −0.5870 |

变化单位为百分点。三个指标均明显下降，说明直接换成 A/B 会失去
YES/NO 的自然语义优势。

## 6. E2：A/B 交换校准

### 做法

每个阈值执行两次 A/B 判断。

正向：

```text
A: severity >= τ
B: severity < τ
```

交换语义：

```text
A: severity < τ
B: severity >= τ
```

异常方向的两个 margin 为：

$$
m_1=\ell_A^{forward}-\ell_B^{forward},
$$

$$
m_2=\ell_B^{swap}-\ell_A^{swap}.
$$

严格执行先平均 margin、再 sigmoid：

$$
p_k=\sigma\left(\frac{m_1+m_2}{2T}\right).
$$

该设计用于抵消模型对 A/B token 或选项位置的固定偏好。

### 效果

| 数据集 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| MSAD | 93.4871% | 79.4267% | 75.7109% |
| 相对 E0 | −0.8759 | −1.7759 | −0.5546 |
| 相对 E1 | +0.2618 | +0.4422 | +0.0325 |

E2 全面高于 E1，证明交换校正确实抵消了一部分 A/B 偏置；但它仍低于 E0，
且文本侧问题数量从 10 增加到 20。

## 7. E3：YES/NO 问题极性反转校准

### 做法

保留 YES/NO 候选，但对每个阈值同时询问正向命题与互补命题。

正向：

```text
Is the anomaly severity at least τ?
```

反向：

```text
Is the anomaly severity strictly below τ?
```

两个异常方向 margin 为：

$$
m_1=\ell_{\mathrm{YES}}^{forward}
    -\ell_{\mathrm{NO}}^{forward},
$$

$$
m_2=\ell_{\mathrm{NO}}^{reverse}
    -\ell_{\mathrm{YES}}^{reverse}.
$$

最终：

$$
p_k=\sigma\left(\frac{m_1+m_2}{2T}\right).
$$

### 效果

| 数据集 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| MSAD | 93.9734% | 80.3436% | 76.1857% |
| 相对 E0 | −0.3896 | −0.8590 | −0.0798 |
| 相对 E2 | +0.4863 | +0.9169 | +0.4748 |

E3 恢复了 E2 的大部分损失，说明 YES/NO 语义比 A/B 更稳定。但是 E3 没有
超过简单 E0，且需要 20 个问题，因此没有性能或效率上的主方法优势。

## 8. E4：中心密集视频帧采样

### 做法

E4 完全保留 E0 的 system prompt、YES/NO 问法、10 个阈值、PAVA 和最终
评分公式，只改变 10 秒窗口内 10 帧的时间位置。

E0 在完整窗口中均匀取 10 帧。E4 改为：

- 4 帧均匀覆盖完整 10 秒窗口，保留全局上下文；
- 6 帧密集覆盖窗口中心区域，提高短动作的采样密度；
- 仍然只向模型输入 10 帧；
- 从 2 FPS 解码候选池中确定性选择帧；
- 很短的边界窗口优先保留全部真实帧，必要时重复最接近中心的帧。

其目标是减少均匀采样遗漏短异常、人—物交互和快速动作关键阶段的概率。

### S1：中心 4 秒

```text
4 帧覆盖完整 10 秒 + 6 帧覆盖中心 4 秒
```

| 数据集 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| MSAD | 94.3493% | **81.2101%** | 76.2402% |
| 相对 E0 | −0.0138 | +0.0074 | −0.0253 |

S1 与 E0 几乎完全持平。PR-AUC 提高 0.0074 个百分点，但 ROC-AUC 和
Max-F1 分别下降 0.0138、0.0253 个百分点，不构成稳定提升。

### S2：中心 2 秒

```text
4 帧覆盖完整 10 秒 + 6 帧覆盖中心 2 秒
```

| 数据集 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| MSAD | 94.2867% | 81.0542% | 76.1688% |
| 相对 E0 | −0.0763 | −0.1484 | −0.0967 |

S2 的三个指标都低于 E0，也低于 S1。中心区域过窄可能导致 6 个密集帧的
时间冗余增加，同时削弱对动作开始、发展和结束阶段的覆盖。

## 9. MSAD 全部消融总表

全部结果使用相同测试集、相同 frame-level 评测代码和 `sigma=10`：

| 方法 | ROC-AUC | PR-AUC | Max-F1 | 相对 E0 计算量 |
|---|---:|---:|---:|---:|
| **E0：YES/NO** | **94.3630%** | 81.2026% | **76.2655%** | 1.0× |
| E1：单向 A/B | 93.2254% | 78.9846% | 75.6784% | 约 1.0× |
| E2：A/B 交换校准 | 93.4871% | 79.4267% | 75.7109% | 文本侧约 2.0× |
| E3：YES/NO 极性反转 | 93.9734% | 80.3436% | 76.1857% | 文本侧约 2.0× |
| E4-S1：中心 4 秒 | 94.3493% | **81.2101%** | 76.2402% | 约 1.0× |
| E4-S2：中心 2 秒 | 94.2867% | 81.0542% | 76.1688% | 约 1.0× |

所有版本均完成：

```text
240/240 个 MSAD 测试视频
9,250/9,250 个 clips
0 个失败窗口
```

## 10. 与作者基线比较

| 数据集 | 方法 | ROC-AUC | PR-AUC | Max-F1 |
|---|---|---:|---:|---:|
| UCF-Crime | VADTree | 84.74% | **41.89%** | 44.29% |
| UCF-Crime | URF-HVAA | 84.36% | 36.15% | 41.83% |
| UCF-Crime | **E0** | **86.27%** | 39.47% | **45.22%** |
| MSAD | VADTree | 89.32% | 71.41% | 68.80% |
| MSAD | URF-HVAA | 93.06% | 77.81% | 74.82% |
| MSAD | **E0** | **94.36%** | **81.20%** | **76.27%** |
| XD-Violence | VADTree | 90.47% | 67.91% | 69.17% |
| XD-Violence | URF-HVAA | 91.34% | 68.07% | 71.93% |
| XD-Violence | **E0** | **92.11%** | **75.36%** | **73.13%** |

其中 E1–E4、S1、S2 目前是 MSAD 上的完整消融，不能把它们写成已经完成
三个数据集的结果。

## 11. 当前方法选择

当前主方法应确定为：

> **E0：基于原始视频的 YES/NO 累计阈值似然异常评分。**

选择理由：

- 三个数据集均已完整验证；
- MSAD 消融中 ROC-AUC 和 Max-F1 最高；
- 不生成 caption、异常标签或 refinement；
- 没有人工融合权重；
- 只需 10 个阈值问题；
- 连续分数具有尾概率期望解释；
- E1–E3 的语言校准和 E4-S1/S2 的采样调整均未稳定超过它。

E4-S1 是最接近 E0 的采样消融，可以作为“输入帧分布对性能影响较小”的
实验结果保留，但不应声称它显著优于 E0。

## 12. 实现和结果位置

| 方法 | 代码 | MSAD 结果目录 |
|---|---|---|
| E0 | `src/video_cumulative_score.py` | `data/MSAD/scores/videollama3_cumulative_likelihood_optimized/` |
| E1 | `src/video_ab_calibrated_score.py` | `data/MSAD/scores/videollama3_cumulative_likelihood_e1_ab/` |
| E2 | `src/video_ab_calibrated_score.py` | `data/MSAD/scores/videollama3_cumulative_likelihood_e2_ab_swap/` |
| E3 | `src/video_yesno_calibrated_score.py` | `data/MSAD/scores/videollama3_cumulative_likelihood_e3_yesno_polarity_swap/` |
| E4-S1 | `src/video_center_dense_score.py` | `data/MSAD/scores/videollama3_cumulative_likelihood_e4_center_dense_g4_c6_s4/` |
| E4-S2 | `src/video_center_dense_score.py` | `data/MSAD/scores/videollama3_cumulative_likelihood_e4_center_dense_g4_c6_s2/` |

统一评测代码为 `src/eval.py`。本文没有引用任何名称包含
`temporal_fusion` 的结果目录。
