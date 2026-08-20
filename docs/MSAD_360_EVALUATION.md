# MSAD 360-Video Evaluation

This report documents the complete MSAD protocol using all 360 test videos.
The 240-video results retained in the original release remain available for
historical comparison, but the tables in this report use one fixed 360-video
index and one fixed temporal annotation file.

## Protocol

- Video index: `results/MSAD/annotations/test.txt` (360 videos)
- Temporal annotations: `results/MSAD/annotations/msad_anomaly_index.txt`
- Metrics: ROC-AUC, PR-AUC, and Max-F1
- Existing score files are resumed rather than overwritten.
- Metrics are written to each experiment's `metrics_360/` directory.

## Reproduction

From the repository root, run the desired method with the current research
launcher. The launcher supports checkpoint recovery and multi-GPU execution:

```bash
METHODS=e0 GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=e1 GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=e2 GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=e3 GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=e4_s4 GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=e4_s2 GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=threshold5 GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=stride32_e0 GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=direct_generated GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=direct_promptfix GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
METHODS=caption_controlled GPU_IDS=0,1,2 bash research_code/scripts/run_msad_360_resume_covas.sh
```

To evaluate every complete score directory without GPU inference:

```bash
bash research_code/scripts/eval_msad_360_all_existing.sh
```

## Output locations

The main output directories are under `data/MSAD/scores/`. Each complete
directory should contain `metrics_360/roc_auc.txt`, `metrics_360/pr_auc.txt`,
and `metrics_360/max_f1.txt`. The release's compact experiment table is in
[`ABLATION_RESULTS.md`](ABLATION_RESULTS.md).

Incomplete directories or directories containing unresolved error records are
not promoted to the formal results table.
