#!/usr/bin/env python3
"""Summarize backbone and prompt ablations without mixing partial runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DATASET_CONFIG = {
    "UCF-Crime": ("data/ucf_crime", "data/ucf_crime/annotations/test.txt"),
    "MSAD": ("data/MSAD", "results/msad/annotations/test.txt"),
    "XD-Violence": ("data/xd_violence", "data/xd_violence/annotations/test.txt"),
}
PROMPT_IDS = ("baseline", "visible_evidence", "reach_level", "no_less_than", "rated_above")
METRICS = ("roc_auc", "pr_auc", "max_f1")
DEFAULT_VIDEOLLAMA_ROOTS = {
    "UCF-Crime": "data/ucf_crime/scores/videollama3_cumulative_likelihood",
    "MSAD": "data/MSAD/scores/videollama3_cumulative_likelihood_optimized",
    "XD-Violence": "data/xd_violence/scores/videollama3_cumulative_likelihood_optimized",
}


def _video_names(index: Path) -> set[str]:
    return {line.split()[0].replace(".mp4", "") for line in index.read_text().splitlines() if line.strip()}


def _metric(root: Path, name: str) -> float | None:
    path = root / "metrics" / f"{name}.txt"
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _scores(root: Path) -> dict[tuple[str, str], float]:
    result = {}
    for path in root.glob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            values = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(values, dict):
            continue
        stem = path.stem
        for frame, score in values.items():
            if isinstance(score, (int, float)) and math.isfinite(float(score)):
                result[(stem, str(frame))] = float(score)
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    try:
        from scipy.stats import spearmanr

        value = float(spearmanr(left, right).statistic)
        return value if math.isfinite(value) else None
    except Exception:
        # A dependency-free rank fallback for environments without SciPy.
        def rank(values):
            order = sorted(range(len(values)), key=values.__getitem__)
            ranks = [0.0] * len(values)
            for position, index in enumerate(order):
                ranks[index] = float(position)
            return ranks

        a, b = rank(left), rank(right)
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den if den else None


def _run_row(dataset: str, model: str, root: Path, expected: set[str]) -> dict[str, Any]:
    observed = {path.stem for path in root.glob("*.json") if not path.name.startswith("_")}
    errors = sum(1 for _ in (root / "_errors").glob("*.json")) if (root / "_errors").is_dir() else 0
    complete = len(expected & observed) == len(expected) and errors == 0
    row: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "root": str(root),
        "expected_videos": len(expected),
        "observed_videos": len(observed),
        "errors": errors,
        "status": "complete" if complete else "partial",
    }
    row.update({metric: _metric(root, metric) for metric in METRICS})
    return row


def _prompt_rows(root: Path, expected: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    values: dict[str, dict[tuple[str, str], float]] = {}
    for prompt_id in PROMPT_IDS:
        run_root = root / prompt_id
        row = _run_row("MSAD", prompt_id, run_root, expected)
        rows.append(row)
        values[prompt_id] = _scores(run_root)
    baseline = values["baseline"]
    correlations = {}
    for prompt_id in PROMPT_IDS:
        if prompt_id == "baseline":
            continue
        keys = sorted(set(baseline) & set(values[prompt_id]))
        left = [baseline[key] for key in keys]
        right = [values[prompt_id][key] for key in keys]
        correlations[prompt_id] = {
            "n_clips": len(keys),
            "spearman": _spearman(left, right),
            "score_mae": sum(abs(a - b) for a, b in zip(left, right)) / len(keys) if keys else None,
        }
    for metric in METRICS:
        measured = [row[metric] for row in rows if row.get("status") == "complete" and isinstance(row[metric], (int, float))]
        if measured:
            mean = sum(measured) / len(measured)
            sample_std = math.sqrt(sum((x - mean) ** 2 for x in measured) / (len(measured) - 1)) if len(measured) > 1 else 0.0
            worst = min(measured)
            span = max(measured) - min(measured)
        else:
            mean = sample_std = worst = span = None
        for row in rows:
            value = row[metric]
            baseline_value = rows[0][metric]
            row[f"{metric}_delta_baseline"] = value - baseline_value if row.get("status") == "complete" and rows[0].get("status") == "complete" and isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)) else None
        correlations[f"summary_{metric}"] = {"mean": mean, "sample_std": sample_std, "worst": worst, "max_min_range": span}
    return rows, correlations


def _write_markdown(path: Path, rows: list[dict[str, Any]], correlations: dict[str, Any], differences: list[dict[str, Any]]) -> None:
    lines = ["# COVAS ablation summary", "", "## Backbone", "", "| Dataset | Model | ROC-AUC | PR-AUC | Max-F1 | Status |", "|---|---|---:|---:|---:|---|"]
    for row in rows:
        if row.get("model") in {"Qwen3-VL-8B", "VideoLLaMA3-7B"}:
            lines.append(f"| {row['dataset']} | {row['model']} | {row.get('roc_auc')} | {row.get('pr_auc')} | {row.get('max_f1')} | {row['status']} |")
    lines += ["", "### Backbone difference (Qwen3-VL minus VideoLLaMA3)", "", "| Dataset | ROC-AUC Δ | PR-AUC Δ | Max-F1 Δ |", "|---|---:|---:|---:|"]
    for row in differences:
        lines.append(f"| {row['dataset']} | {row.get('roc_auc_delta')} | {row.get('pr_auc_delta')} | {row.get('max_f1_delta')} |")
    lines += ["", "## Prompt sensitivity (MSAD)", "", "| Prompt | ROC-AUC | PR-AUC | Max-F1 | Status |", "|---|---:|---:|---:|---|"]
    for row in rows:
        if row.get("dataset") == "MSAD" and row.get("model") in PROMPT_IDS:
            lines.append(f"| {row['model']} | {row.get('roc_auc')} | {row.get('pr_auc')} | {row.get('max_f1')} | {row['status']} |")
    lines += ["", "### Prompt score agreement", "", "| Prompt | Clips | Spearman | Score MAE |", "|---|---:|---:|---:|"]
    for prompt_id in PROMPT_IDS:
        if prompt_id in correlations:
            item = correlations[prompt_id]
            lines.append(f"| {prompt_id} | {item['n_clips']} | {item['spearman']} | {item['score_mae']} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("data/ablation_summary"))
    parser.add_argument("--videollama_roots", type=Path, default=None, help="Optional JSON mapping dataset to existing VideoLLaMA3 score roots.")
    args = parser.parse_args()
    root = args.root
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    video_roots = json.loads(args.videollama_roots.read_text()) if args.videollama_roots else DEFAULT_VIDEOLLAMA_ROOTS
    rows: list[dict[str, Any]] = []
    for dataset, (dataset_dir, index_file) in DATASET_CONFIG.items():
        expected = _video_names(root / index_file)
        qwen_root = root / dataset_dir / "scores/covas_qwen3vl_8b_baseline"
        rows.append(_run_row(dataset, "Qwen3-VL-8B", qwen_root, expected))
        if dataset in video_roots:
            rows.append(_run_row(dataset, "VideoLLaMA3-7B", root / video_roots[dataset], expected))
    backbone_rows = list(rows)
    prompt_expected = _video_names(root / DATASET_CONFIG["MSAD"][1])
    prompt_rows, prompt_stats = _prompt_rows(root / "data/MSAD/scores/covas_prompt_sensitivity", prompt_expected)
    rows.extend(prompt_rows)
    differences = []
    for dataset in DATASET_CONFIG:
        qwen = next((row for row in rows if row.get("dataset") == dataset and row.get("model") == "Qwen3-VL-8B"), None)
        llama = next((row for row in rows if row.get("dataset") == dataset and row.get("model") == "VideoLLaMA3-7B"), None)
        if qwen and llama:
            differences.append({
                "dataset": dataset,
                **{f"{metric}_delta": qwen[metric] - llama[metric] if qwen.get("status") == "complete" and llama.get("status") == "complete" and isinstance(qwen[metric], (int, float)) and isinstance(llama[metric], (int, float)) else None for metric in METRICS},
            })
    payload = {"backbone": backbone_rows, "backbone_differences": differences, "prompt_sensitivity": prompt_rows, "prompt_statistics": prompt_stats}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with (output / "summary.csv").open("w", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(output / "summary.md", rows, prompt_stats, differences)
    print(f"wrote {output / 'summary.csv'}, {output / 'summary.json'}, {output / 'summary.md'}")


if __name__ == "__main__":
    main()
