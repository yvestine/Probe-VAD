# Probe-VAD

**Ordinal Likelihood Probing for Training-Free Video Anomaly Detection**

**Official implementation of [Probe-VAD: Ordinal Likelihood Probing for Training-Free Video Anomaly Detection](https://arxiv.org/abs/2609.17211).**

Probe-VAD is a training-free video anomaly detection framework built on
VideoLLaMA3. It directly scores raw video clips through ordinal likelihood
probing, without generating captions, anomaly tags, or refinement text.

This repository provides the official implementation of Probe-VAD together
with the code, evaluation scripts, precomputed scores, and experimental
results used in the paper. Additional ablations and analysis utilities are
included under `research_code/` and `results/experiments/` for reproducibility
and further study.

## Highlights

- Direct visual scoring with no caption bottleneck.
- Continuous ordinal score from ten cumulative likelihood questions.
- One video decode and one visual encoding shared by all thresholds.
- Batched threshold inference and optional language-model prefix KV cache.
- PAVA monotonic projection with a probabilistic expectation interpretation.
- Atomic per-video checkpoints and clip-level recovery after interruption.
- Single-GPU and LPT-balanced multi-GPU execution.
- Compatible evaluation for UCF-Crime, MSAD, and XD-Violence annotations.

## Method

For each video, Probe-VAD places a fixed 10-second window at center frames
`0, 16, 32, ...`. The window is sampled at 2 FPS with at most 10 RGB frames.
VideoLLaMA3 then answers ten ordered questions:

```text
Is the anomaly severity visible in this video segment at least 0.1?
...
Is the anomaly severity visible in this video segment at least 1.0?
```

At threshold $\tau_k=k/10$, the `YES` and `NO` logits define

$$
p_k=P(S\ge\tau_k\mid V)
   =\mathrm{softmax}(\ell_{\mathrm{NO}},\ell_{\mathrm{YES}})_{\mathrm{YES}}.
$$

PAVA projects the ten tail probabilities onto
$p_1 \ge p_2 \ge \cdots \ge p_{10}$. 
The final anomaly score is

$$
s(V)=0.1\sum_{k=1}^{10}p_k
    =\frac{1}{10}\sum_{k=1}^{10}p_k.
$$

See [docs/METHOD.md](docs/METHOD.md) for the exact prompt, window definition,
optimized forward path, checkpoint policy, and evaluation protocol.

## Repository layout

```text
.
├── probe_vad/
│   ├── scoring.py          # raw-video scoring
│   ├── utils.py            # video, likelihood, PAVA, checkpoint utilities
│   ├── evaluation.py       # frame-level ROC/PR/F1 evaluation
│   └── video_record.py
├── configs/                # UCF-Crime, MSAD, XD-Violence templates
├── docs/                   # method and dataset documentation
├── examples/               # tiny score/evaluation format example
├── research_code/          # research sources, ablation launchers, old baseline
├── results/                # scores plus preserved experiment outputs
├── scripts/                # config and balanced multi-GPU launchers
├── tests/                  # CPU mathematical and interface tests
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

The release includes all completed score JSON files, metrics, exact
evaluation annotations, and the selected research score archives. Partial
checkpoint directories are marked in `results/EXPERIMENT_MANIFEST.json` and
are not promoted as full-test results. It does not include model weights,
dataset videos, extracted frames, or caches. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the experiment index and
author-baseline comparison.

## Installation

Python 3.10 is recommended.

### Conda

```bash
conda env create -f environment.yml
conda activate probe-vad
pip install -e .
```

### Existing CUDA/PyTorch environment

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

VideoLLaMA3 normally uses FlashAttention 2 on CUDA. Install a `flash-attn`
wheel/build compatible with your PyTorch, CUDA, GPU, and compiler:

```bash
pip install flash-attn --no-build-isolation
```

If FlashAttention is unavailable, pass `--attn_implementation eager` to the
single-GPU scorer. The multi-GPU launcher uses FlashAttention by default.

## Model

The default model is:

```text
DAMO-NLP-SG/VideoLLaMA3-7B
```

On the first online run, Transformers may download the model and its trusted
remote code. To use a local checkpoint and prevent network access:

```bash
export MODEL_PATH=/absolute/path/to/VideoLLaMA3-7B
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Review and accept the model's upstream license before use.

## Dataset preparation

Use this layout:

```text
data/<dataset>/
├── videos/
│   ├── video_001.mp4
│   └── ...
├── frames/                 # paths are used by the evaluation metadata
└── annotations/
    ├── anomaly_test.txt
    └── temporal_annotations.txt
```

Raw videos are required for scoring. Extracted frame images are not read by the
current evaluator, but the frame root remains part of the URF/LAVAD-compatible
evaluation interface.

Annotation formats and dataset-specific naming notes are documented in
[docs/DATASETS.md](docs/DATASETS.md).

Validate paths and metadata before using a GPU:

```bash
probe-validate \
  --video_dir ./data/MSAD/videos \
  --index_file ./results/msad/annotations/test.txt \
  --temporal_annotation_file ./results/msad/annotations/temporal_annotations.txt
```

## Quick start

Copy and edit one configuration:

```bash
cp configs/msad.env.example configs/msad.env
```

Run the complete scoring-to-metrics pipeline:

```bash
bash scripts/run_from_config.sh configs/msad.env
```

For several GPUs:

```bash
sed -i 's/^GPU_IDS=.*/GPU_IDS=0,2,6/' configs/msad.env
bash scripts/run_from_config.sh configs/msad.env
```

The launcher scans existing outputs, computes the number of unfinished clips
per video, assigns remaining work using longest-processing-time balancing, and
starts one worker per GPU entry. Repeating a GPU ID intentionally starts
multiple workers on that GPU, subject to memory capacity.

## Single-GPU scoring

After `pip install -e .`:

```bash
CUDA_VISIBLE_DEVICES=0 probe-score \
  --video_dir ./data/MSAD/videos \
  --index_file ./results/msad/annotations/test.txt \
  --output_dir ./data/MSAD/scores/probe_vad \
  --model_path DAMO-NLP-SG/VideoLLaMA3-7B \
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

Use `probe-score --help` for all arguments.

## Output and recovery

Each video produces one JSON file:

```json
{
  "0": 0.1032,
  "16": 0.1176,
  "32": 0.6814
}
```

Keys are center-frame indices in the original video. Completed clips are saved
atomically every `CHECKPOINT_INTERVAL` windows. A restart with `--resume`
retains valid scores and retries only failed/missing clips.

The source runner may create temporary failure/checkpoint diagnostics while it
is running, but the released result archive keeps only final score JSON files
and compact metrics.

## Evaluation

The multi-GPU launcher evaluates automatically. To evaluate an existing score
directory:

```bash
probe-eval \
  --root_path ./data/MSAD/frames \
  --annotationfile_path ./results/msad/annotations/test.txt \
  --temporal_annotation_file ./results/msad/annotations/temporal_annotations.txt \
  --scores_dir ./data/MSAD/scores/probe_vad \
  --output_dir ./data/MSAD/scores/probe_vad/metrics \
  --frame_interval 16 \
  --normal_label 0
```

The evaluator numerically sorts center-frame keys, smooths the clip sequence
with a Gaussian kernel (`sigma=10` by default), repeats every score for 16
frames, aligns it to the annotated frame length, and reports ROC-AUC, PR-AUC,
and Max-F1.

All bundled results can be evaluated immediately without downloading the model
or videos:

```bash
bash scripts/evaluate_precomputed.sh all
```

## Tests

CPU-only checks:

```bash
pip install -e '.[dev]'
pytest
bash -n scripts/run_probe_vad_rebalanced.sh
bash -n scripts/run_from_config.sh
```

An evaluator smoke test that needs no real video:

```bash
probe-eval \
  --root_path ./examples/frames \
  --annotationfile_path ./examples/annotations/test.txt \
  --temporal_annotation_file ./examples/annotations/temporal.txt \
  --scores_dir ./examples/scores \
  --output_dir /tmp/probe_demo_metrics \
  --frame_interval 16 \
  --normal_label 0 \
  --no_smoothing
```

## Main results

The following results correspond to the evaluation protocols reported in the
current Probe-VAD paper.

| Dataset | ROC-AUC (%) | PR-AUC (%) |
|---|---:|---:|
| UCF-Crime | **86.27** | 39.47 |
| MSAD | **87.55** | **78.43** |
| XD-Violence | **92.11** | **75.36** |

For MSAD, the paper uses the complete 360-video test protocol comprising
14,372 clips. The earlier 240-video MSAD result included in the original
repository release is retained only for historical reproducibility and should
not be interpreted as the paper's main MSAD result.

Exact MSAD-360 metrics and protocol details are available in
[`results/MSAD_360_METRICS.json`](results/MSAD_360_METRICS.json) and
[`docs/MSAD_360_EVALUATION.md`](docs/MSAD_360_EVALUATION.md).



## Supplementary experiments

Additional experiments accompanying the current paper are summarized in
[`docs/SUPPLEMENTARY_EXPERIMENTS.md`](docs/SUPPLEMENTARY_EXPERIMENTS.md).

The supplementary analyses include:

- sensitivity to the number of ordinal thresholds;
- uniform and non-uniform severity-threshold placement;
- threshold-wise and severity-band response analysis;
- clip-internal frame-sampling sensitivity;
- temporal-smoothing sensitivity;
- end-to-end runtime and intermediate-storage efficiency.

These experiments do not change the default Probe-VAD configuration:
10 uniformly spaced severity thresholds, uniform frame sampling, and
Gaussian smoothing with `sigma=10`.

## Scope and limitations

- This is a training-free inference method, not a trained anomaly classifier.
- Scores inherit the visual, cultural, and domain biases of the base VLM.
- The public-surveillance prompt may not transfer unchanged to other domains.
- The method is intended for research; it must not be used as the sole basis
  for safety, policing, employment, or other high-impact decisions.
- Dataset results depend on exact video versions, annotations, decoding, and
  evaluation parameters.

## Citation and acknowledgements

Citation metadata is provided in [CITATION.cff](CITATION.cff). Please also cite
VideoLLaMA3 and the upstream URF-HVAA work when applicable.

## License

The Probe-VAD source in this repository is released under the
[MIT License](LICENSE). Third-party models, datasets, and dependencies retain
their own licenses; see [NOTICE](NOTICE).
