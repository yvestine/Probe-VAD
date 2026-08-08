# COVAS-VAD

**COVAS-VAD：Cumulative Ordinal Visual Anomaly Scoring for Video Anomaly
Detection（累计有序视觉异常评分）**

这是从 URF-HVAA 实验仓库中独立整理出的 COVAS-VAD 开源项目。顶层
`covas_vad/` 是可直接运行的 E0 主方法；`research_code/` 和
`results/experiments/` 则保留本研究已经完成或实际运行过的消融代码、运行入口、
分数 JSON 和评测记录，便于复现、审计和后续扩展。主方法本身不依赖 caption、
异常标签、refinement 或 score fusion。

## 方法

每个视频按中心帧构造固定长度滑动窗口。默认设置为：

- 窗口长度：10 秒；
- 中心帧步长：16 原视频帧，相邻窗口通常重叠；
- 视频采样：2 FPS，最多输入 10 张均匀采样 RGB 帧；
- 视觉语言模型：VideoLLaMA3-7B；
- 输入：当前原始视频窗口，不生成任何中间文本表征。

对每个窗口依次判断十个有序命题：

```text
Is the anomaly severity visible in this video segment at least 0.1?
...
Is the anomaly severity visible in this video segment at least 1.0?
```

模型不自由生成分数，而是在每个阈值处读取单 token 候选 `YES` 与 `NO` 的
logits。阈值尾概率为：

```text
p_k = softmax(logit(NO), logit(YES))[YES]
    ≈ P(S >= k/10 | video),  k=1,...,10
```

十个问题共享同一次视频解码和视觉编码，并批量完成文本推理。随后使用 PAVA
将尾概率投影为非递增序列：

```text
p_1 >= p_2 >= ... >= p_10
```

最终连续异常分数由尾概率积分的离散近似得到：

```text
score = 0.1 * sum(p_k) = mean(p_1,...,p_10)
```

输出 JSON 的键是原视频中心帧编号，值是该窗口的异常分数：

```json
{"0": 0.1032, "16": 0.1176, "32": 0.6814}
```

评测保持原接口：clip 分数先在 clip 序列上进行高斯平滑（默认 sigma=10），
再按 16 帧展开为 frame-level 分数，计算 ROC-AUC、PR-AUC 和 Max-F1。

## 目录

```text
COVAS-VAD/
├── covas_vad/
│   ├── scoring.py
│   ├── utils.py
│   ├── evaluation.py
│   └── video_record.py
├── configs/
├── docs/
├── research_code/           # 研究阶段源码、消融脚本和旧 baseline（归档）
├── results/                 # E0完整结果、消融结果、指标和评测标注
├── scripts/
├── tests/
├── README.md
├── README_CN.md
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

`results/` 包含三个数据集的完整 E0 score JSON、metrics、精确评测标注，以及
`results/experiments/` 下已保存的生成式/候选 likelihood、Caption-E0、Stride32、
阈值数、E1–E4、时间顺序、中心密集采样、选择性核查、窗口后处理、标签
refinement 等实验的最终 score JSON。`research_code/` 中的代码按主流程和旧
baseline 分层保存。模型权重、原始视频、抽帧和缓存不打包。

## 环境

推荐创建独立环境并以 editable 方式安装：

```bash
conda env create -f environment.yml
conda activate covas-vad
pip install -e .
```

`flash-attn` 与 CUDA/PyTorch 版本强相关，请根据服务器环境另行安装。如果
模型已缓存在本机，可以通过 `MODEL_PATH` 指向本地模型目录，并设置
`HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。

## 多 GPU 一键运行并评测

推荐复制数据集配置模板：

```bash
cp configs/msad.env.example configs/msad.env
# 修改 configs/msad.env 中的数据、模型和 GPU 路径
bash scripts/run_from_config.sh configs/msad.env
```

脚本会按剩余 clip 数量进行 LPT 均衡分片，在各 GPU 上断点续跑，全部完成后
自动评测并把三项指标打印到终端。常用环境变量：

- `GPU_IDS`：逗号分隔的物理 GPU 编号；
- `MODEL_PATH`：Hugging Face 模型 ID 或本地模型目录；
- `WINDOW_SECONDS`、`FRAME_INTERVAL`、`SAMPLE_FPS`、`MAX_FRAMES`；
- `THRESHOLD_BATCH_SIZE`：一次并行处理的阈值数，显存不足时可从 10 降为 5；
- `PREFIX_CACHE=1`：复用共同文本前缀 KV cache；
- `CHECKPOINT_INTERVAL`：成功处理多少个窗口后原子保存；
- `PRECISE_TIME=1`：用于存在非零流起始时间的视频，MSAD 建议开启。

数据集对应的 `NORMAL_LABEL` 取决于 annotation 文件的标签定义。此前实验使用：
UCF-Crime 为 7、MSAD 为 0、XD-Violence 为 4。

## 单 GPU 仅评分

```bash
CUDA_VISIBLE_DEVICES=0 covas-score \
  --video_dir ./data/MSAD/videos \
  --index_file ./results/msad/annotations/test.txt \
  --output_dir ./data/MSAD/scores/covas_vad \
  --model_path /path/to/VideoLLaMA3-7B \
  --device cuda:0 \
  --frame_interval 16 \
  --window_seconds 10 \
  --sample_fps 2 \
  --max_frames 10 \
  --precise_time \
  --threshold_batch_size 10 \
  --prefix_cache \
  --monotonic_projection \
  --resume
```

## 单独评测

```bash
covas-eval \
  --root_path ./data/MSAD/frames \
  --annotationfile_path ./results/msad/annotations/test.txt \
  --temporal_annotation_file ./results/msad/annotations/temporal_annotations.txt \
  --scores_dir ./data/MSAD/scores/covas_vad \
  --output_dir ./data/MSAD/scores/covas_vad/metrics \
  --frame_interval 16 \
  --normal_label 0
```

## 已有 E0 结果

在原仓库既有运行和同一评测接口下：

| 数据集 | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| UCF-Crime | 86.27% | 39.47% | 45.22% |
| MSAD | 94.36% | 81.20% | 76.27% |
| XD-Violence | 92.11% | 75.36% | 73.13% |

仓库已包含对应的完整 score JSON 和 metrics。无需模型或 GPU 即可复算：

```bash
bash scripts/evaluate_precomputed.sh all
```

结果文件数量和精确指标见 `results/MANIFEST.json`；所有研究实验的路径、完整性
状态和已报告指标见 `results/EXPERIMENT_MANIFEST.json`，汇总说明见
`docs/EXPERIMENTS_CN.md`。

## 研究实验归档

`research_code/src/` 保存 E0、直接生成分数、11 类候选 likelihood、Caption-E0、
E1/E2/E3、中心密集采样、Stride32、时间顺序和公共评分工具；
`research_code/analysis/` 保存离线投影/不确定性、选择性中心核查、SDEE 和
跨窗口后处理代码；`research_code/legacy_baseline/` 保存原 URF-HVAA 的
caption-based baseline 入口以及标签 refinement 代码，作为历史对照而不是
COVAS-VAD 主方法。

结果归档中完整数据集结果与部分断点结果会明确区分。只有标记为 `complete`
的目录用于三数据集主表；`partial` 目录仍保留已经产生的 JSON，不能直接当作
完整测试集指标。论文中的作者对比只使用 URF-HVAA 和 VADTree 原作者公开
指标，不把本仓库的作者实现或人工 fusion 当作基线。

## 开源边界

项目代码采用 MIT License。VideoLLaMA3 权重、数据集视频、CUDA、FFmpeg
和第三方依赖不随本仓库分发；精确评测标注副本随结果提供，但仍受上游许可
约束。使用者需要遵守各自许可证。
完整英文说明见 [README.md](README.md)，方法定义见
[docs/METHOD.md](docs/METHOD.md)，数据格式见 [docs/DATASETS.md](docs/DATASETS.md)。
