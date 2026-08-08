# Bundled E0 results and experiment archive

This directory contains the complete COVAS-VAD E0 score JSON files used for
the reported results. `experiments/` contains the preserved score JSON and
metrics from the research ablations; it is intentionally separate from the
canonical E0 directories.

| Dataset | Videos | Clips | Score JSON | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime | 290 | 69,634 | 290 | 86.27% | 39.47% | 45.22% |
| MSAD | 240 | 9,250 | 240 | 94.36% | 81.20% | 76.27% |
| XD-Violence | 800 | 146,449 | 800 | 92.11% | 75.36% | 73.13% |

Structure:

```text
results/<dataset>/
├── annotations/
│   ├── test.txt
│   └── temporal_annotations.txt
├── scores/
│   └── <video_id>.json
└── metrics/
    ├── roc_auc.txt
    ├── pr_auc.txt
    ├── max_f1.txt
    └── optimal_thresholds.txt
```

The release intentionally keeps only final score JSON files and compact metrics.
Raw tail-probability dumps, PAVA diagnostics, motion matrices, temporary error
shards, and checkpoint timing files are not part of the package.

Run all bundled evaluations without a model or GPU:

```bash
bash scripts/evaluate_precomputed.sh all
```

The score JSON files are complete for the included test indices. They contain
exactly 69,634 UCF-Crime clips, 9,250 MSAD clips, and 146,449 XD-Violence
clips. `SHA256SUMS` records the integrity of all result and annotation files.

The annotations remain subject to the corresponding dataset/upstream
licenses; see the project `NOTICE`.

The experiment index is `EXPERIMENT_MANIFEST.json`. It distinguishes
`complete` full-test results from `partial` checkpoint directories. See
`../docs/EXPERIMENTS.md` for the method definitions and the author-baseline
comparison table.
