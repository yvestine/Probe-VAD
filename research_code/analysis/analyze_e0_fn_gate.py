"""CPU-only analysis of potential false-negative gates for existing E0 scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data.video_record import VideoRecord
from src.eval import get_video_labels, temporal_testing_annotations
from src.eval_overlap_projection_uncertainty import (
    clip_uncertainty,
    metric_triplet,
    ordered_scores,
    repeat_interval_projection,
)
from src.video_score_utils import (
    atomic_write_json,
    load_json_dict,
    output_stem,
)

GATE_NAMES = (
    "score",
    "tail_entropy",
    "pava_l1",
    "residual_entropy",
    "score_plus_residual_entropy",
)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def binned_residual(
    score: np.ndarray,
    entropy: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """Estimate H-E[H|s] with equal-frequency score bins."""
    if bins <= 1:
        raise ValueError("bins must be greater than one")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(score, quantiles))
    if len(edges) <= 2:
        expected = np.full_like(entropy, float(np.mean(entropy)))
        return entropy - expected, expected, [
            {
                "bin": 0,
                "score_min": float(np.min(score)),
                "score_max": float(np.max(score)),
                "count": int(len(score)),
                "mean_entropy": float(np.mean(entropy)),
            }
        ]
    bin_ids = np.searchsorted(edges[1:-1], score, side="right")
    expected = np.empty_like(entropy)
    summary: list[dict[str, float | int]] = []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        mean_entropy = float(np.mean(entropy[mask]))
        expected[mask] = mean_entropy
        summary.append(
            {
                "bin": bin_id,
                "score_min": float(np.min(score[mask])),
                "score_max": float(np.max(score[mask])),
                "count": int(mask.sum()),
                "mean_entropy": mean_entropy,
            }
        )
    return entropy - expected, expected, summary


def safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, values))
    except ValueError:
        return float("nan")


def top_budget_metrics(
    labels: np.ndarray,
    values: np.ndarray,
    budget: float,
) -> dict[str, float | int]:
    count = max(1, int(np.ceil(len(values) * budget)))
    # Stable sorting keeps the result deterministic when scores are tied.
    selected = np.argsort(-values, kind="stable")[:count]
    selected_fn = int(labels[selected].sum())
    total_fn = int(labels.sum())
    error_rate = float(selected_fn / count)
    prevalence = float(total_fn / len(labels))
    return {
        "audit_fraction": float(count / len(values)),
        "audited_clips": count,
        "covered_fn_clips": selected_fn,
        "total_fn_clips": total_fn,
        "fn_recall": float(selected_fn / total_fn) if total_fn else float("nan"),
        "audit_clip_error_rate": error_rate,
        "error_enrichment": (
            error_rate / prevalence if prevalence > 0 else float("nan")
        ),
    }


def write_budget_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "gate",
                "requested_budget",
                "audit_fraction",
                "audited_clips",
                "covered_fn_clips",
                "total_fn_clips",
                "fn_recall",
                "audit_clip_error_rate",
                "error_enrichment",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--annotation_file", type=Path, required=True)
    parser.add_argument("--temporal_annotation_file", type=Path, required=True)
    parser.add_argument("--scores_dir", type=Path, required=True)
    parser.add_argument("--threshold_details_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--normal_label", type=int, required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--smoothing_sigma", type=float, default=10.0)
    parser.add_argument("--residual_bins", type=int, default=20)
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.05, 0.10, 0.20))
    args = parser.parse_args()
    if args.residual_bins <= 1:
        parser.error("--residual_bins must be greater than one")
    if any(not 0 < value <= 1 for value in args.budgets):
        parser.error("--budgets must lie in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = args.threshold_details_dir or args.scores_dir / "_threshold_details"
    annotations = temporal_testing_annotations(args.temporal_annotation_file)
    records = [
        VideoRecord(line.strip().split(), args.root_path)
        for line in args.annotation_file.read_text().splitlines()
        if line.strip()
    ]

    # First recover the exact frame-level Max-F1 threshold used only for this
    # labeled diagnostic.
    frame_labels: list[np.ndarray] = []
    frame_scores: list[np.ndarray] = []
    videos: list[dict[str, Any]] = []
    for record in tqdm(records, desc="load E0 clips", unit="video"):
        name = Path(record.path).name.removesuffix(".mp4")
        stem = output_stem(name)
        raw_scores = load_json_dict(args.scores_dir / f"{stem}.json")
        details = load_json_dict(detail_dir / f"{stem}.json")
        if not raw_scores:
            raise FileNotFoundError(f"missing score JSON for {stem}")
        if not details:
            raise FileNotFoundError(f"missing threshold details for {stem}")
        centers, raw = ordered_scores(raw_scores)
        if any(str(center) not in details for center in centers):
            raise ValueError(f"incomplete threshold details for {stem}")
        labels = np.asarray(
            get_video_labels(record, annotations, args.normal_label),
            dtype=np.int64,
        ) != args.normal_label
        smoothed = gaussian_filter1d(raw, sigma=args.smoothing_sigma)
        projected = repeat_interval_projection(
            raw,
            len(labels),
            args.frame_interval,
            args.smoothing_sigma,
        )
        frame_labels.append(labels)
        frame_scores.append(projected)
        videos.append(
            {
                "stem": stem,
                "centers": centers,
                "smoothed_scores": smoothed,
                "details": details,
                "frame_labels": labels,
            }
        )

    flat_frame_labels = np.concatenate(frame_labels)
    flat_frame_scores = np.concatenate(frame_scores)
    baseline_metrics = metric_triplet(flat_frame_labels, flat_frame_scores)
    analysis_threshold = baseline_metrics["max_f1_threshold"]

    candidate_rows: list[dict[str, Any]] = []
    for video in videos:
        labels = video["frame_labels"]
        for index, (center, score) in enumerate(
            zip(video["centers"], video["smoothed_scores"])
        ):
            if score >= analysis_threshold:
                continue
            start = index * args.frame_interval
            stop = min(len(labels), start + args.frame_interval)
            if stop <= start:
                continue
            # A clip is an FN target when the majority of its evaluation block
            # is anomalous but E0 predicts it as normal.
            anomaly_fraction = float(np.mean(labels[start:stop]))
            diagnostic = clip_uncertainty(video["details"][str(center)])
            candidate_rows.append(
                {
                    "video": video["stem"],
                    "center_frame": int(center),
                    "score": float(score),
                    "tail_entropy": diagnostic["tail_binary_entropy"],
                    "pava_l1": diagnostic["pava_l1"],
                    "anomaly_fraction": anomaly_fraction,
                    "is_fn": anomaly_fraction >= 0.5,
                }
            )

    score = np.asarray([row["score"] for row in candidate_rows], dtype=np.float64)
    entropy = np.asarray(
        [row["tail_entropy"] for row in candidate_rows],
        dtype=np.float64,
    )
    pava = np.asarray([row["pava_l1"] for row in candidate_rows], dtype=np.float64)
    labels = np.asarray([row["is_fn"] for row in candidate_rows], dtype=bool)
    residual, expected_entropy, bin_summary = binned_residual(
        score,
        entropy,
        args.residual_bins,
    )
    score_rank = percentile_rank(score)
    residual_rank = percentile_rank(residual)
    joint = (score_rank + residual_rank) / 2.0
    gate_values = {
        "score": score,
        "tail_entropy": entropy,
        "pava_l1": pava,
        "residual_entropy": residual,
        "score_plus_residual_entropy": joint,
    }
    for row, expected, residual_value, joint_value in zip(
        candidate_rows,
        expected_entropy,
        residual,
        joint,
    ):
        row["expected_entropy_given_score_bin"] = float(expected)
        row["residual_entropy"] = float(residual_value)
        row["score_plus_residual_entropy"] = float(joint_value)

    aucs = {name: safe_auc(labels, values) for name, values in gate_values.items()}
    correlations = {
        "score_vs_tail_entropy": float(spearmanr(score, entropy).correlation),
        "score_vs_pava_l1": float(spearmanr(score, pava).correlation),
        "score_vs_residual_entropy": float(spearmanr(score, residual).correlation),
    }
    budget_rows: list[dict[str, Any]] = []
    nested_budget_results: dict[str, dict[str, Any]] = {}
    for gate in GATE_NAMES:
        nested_budget_results[gate] = {}
        for budget in args.budgets:
            result = top_budget_metrics(labels, gate_values[gate], budget)
            key = f"{budget:.4f}"
            nested_budget_results[gate][key] = result
            budget_rows.append(
                {
                    "gate": gate,
                    "requested_budget": budget,
                    **result,
                }
            )

    summary = {
        "definition": {
            "prediction_score": "E0 clip scores after Gaussian sigma=10",
            "predicted_normal": f"score < labeled-analysis threshold {analysis_threshold}",
            "clip_ground_truth": (
                "FN when at least 50% of the corresponding repeat-16 frame block "
                "is anomalous"
            ),
            "residual_entropy": (
                f"H - mean(H) within {args.residual_bins} equal-frequency score bins"
            ),
            "joint_gate": "mean of global percentile-rank(score) and rank(H_res)",
        },
        "baseline_frame_metrics": baseline_metrics,
        "candidate_clips": int(len(labels)),
        "fn_clips": int(labels.sum()),
        "fn_prevalence": float(np.mean(labels)),
        "gate_fn_vs_tn_roc_auc": aucs,
        "gate_correlations": correlations,
        "score_bin_summary": bin_summary,
        "budget_results": nested_budget_results,
    }
    atomic_write_json(args.output_dir / "fn_gate_analysis.json", summary)
    atomic_write_json(args.output_dir / "candidate_clips.json", {
        f"{row['video']}::{row['center_frame']}": row for row in candidate_rows
    })
    write_budget_csv(args.output_dir / "fn_gate_budget_results.csv", budget_rows)

    print("\nPotential-FN gate analysis")
    print(f"Analysis-only Max-F1 threshold: {analysis_threshold:.9f}")
    print(
        f"Predicted-normal candidates: {len(labels)}, "
        f"FN clips: {int(labels.sum())} ({np.mean(labels) * 100:.4f}%)"
    )
    print("\nGate FN-vs-TN ROC-AUC")
    for gate in GATE_NAMES:
        print(f"{gate:30s} {aucs[gate] * 100:8.4f}%")
    print("\nSpearman correlations")
    for name, value in correlations.items():
        print(f"{name:30s} {value:8.4f}")
    print("\nFixed audit budgets")
    print("gate                           budget  FN-recall  audit-error  lift")
    for row in budget_rows:
        print(
            f"{row['gate']:30s} "
            f"{row['requested_budget'] * 100:5.1f}% "
            f"{row['fn_recall'] * 100:9.4f}% "
            f"{row['audit_clip_error_rate'] * 100:11.4f}% "
            f"{row['error_enrichment']:6.3f}"
        )
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
