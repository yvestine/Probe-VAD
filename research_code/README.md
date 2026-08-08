# Research Code Archive

This directory contains the research-stage source code and launchers used to
evaluate COVAS-VAD and its comparison experiments. It is not a second
installable package.

- `src/`: E0 and E1/E2/E3/E4 scoring code, stride ablations, direct generation,
  candidate likelihood, caption-E0, temporal-order experiments, and shared
  evaluation utilities.
- `analysis/`: offline aggregation, uncertainty diagnostics, selective-center
  checks, temporal post-processing, and SDEE pilot code.
- `scripts/`: multi-GPU and resumable experiment launchers.
- `legacy_baseline/`: historical caption, Llama, scoring, tag, and refinement
  entry points. These are retained for audit and are not part of the COVAS-VAD
  primary method.

Run research scripts from the repository root and add this directory to the
Python path:

```bash
cd /path/to/COVAS-VAD
export PYTHONPATH="$PWD/research_code:$PYTHONPATH"
```

Scripts keep model, video, annotation, and GPU paths configurable. Model
weights, videos, caches, and temporary outputs are not bundled. Some historical
baseline scripts additionally require external VADTree outputs or legacy model
environments. Experiment paths, metrics, and `complete`/`partial` status are
listed in `results/EXPERIMENT_MANIFEST.json`.
