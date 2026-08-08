# 研究代码归档

这里保存的是 URF-HVAA 研究阶段实际使用过的代码和入口，不是另一个独立的
安装包。它与顶层 `covas_vad/` 的关系如下：

- `src/`：E0 主流程和 E1/E2/E3、E4、Stride32、直接生成/候选 likelihood、
  Caption-E0、时间顺序消融等实际评分代码；
- `analysis/`：离线阈值聚合、PAVA/熵分析、选择性中心核查、跨窗口后处理、
  SDEE-D1 pilot 代码；
- `scripts/`：对应的多 GPU/断点续跑入口；
- `legacy_baseline/`：原始 caption → Llama → score → tag/refinement 代码，
  只为历史对照和复现保留，不属于 COVAS-VAD 主方法。

运行这些脚本时应在包含数据集的工作区执行，并把本目录的源码加入
`PYTHONPATH`，例如：

```bash
cd /path/to/COVAS-VAD
export PYTHONPATH="$PWD/research_code:$PYTHONPATH"
```

脚本中的模型、视频、annotation 和 GPU 路径均保留为实验时的可配置参数；
模型权重、视频和缓存没有打包。部分历史 baseline 还需要外部的 VADTree
结果或旧版模型环境，这些依赖不属于 COVAS-VAD 的主方法安装路径。每个实验
结果对应的打包路径、视频数、metrics 和 `complete/partial` 状态见
`../results/EXPERIMENT_MANIFEST.json`。
