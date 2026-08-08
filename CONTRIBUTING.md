# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
conda env create -f environment.yml
conda activate covas-vad
pip install -e '.[dev]'
pytest
```

Keep changes scoped to the direct cumulative ordinal scoring pipeline. New
ablations should use separate modules and output directories so the reference
E0 implementation remains reproducible.

Before submitting a change:

```bash
ruff check covas_vad tests
pytest
bash -n scripts/run_covas_vad_rebalanced.sh
bash -n scripts/run_from_config.sh
```

Do not commit model weights, dataset media, private paths, caches, generated
score JSON files, or credentials.

