"""Evaluate overlap-aware clip-to-frame projection and E0 uncertainty.

This is a CPU-only diagnostic over already generated E0 score JSON files and,
when available, their ``_threshold_details``.  It never loads VideoLLaMA3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from src.data.video_record import VideoRecord
from src.eval import get_video_labels, temporal_testing_annotations
from src.video_score_utils import (
    atomic_write_json,
    get_video_info,
    load_json_dict,
    output_stem,
    resolve_video_path,
)

EPSILON = 1e-12
PROJECTION_NAMES = ("repeat16_sigma10", "overlap_mean", "overlap_max", "overlap_top3")
UNCERTAINTY_NAMES = (
    "tail_binary_entropy",
    "ordinal_entropy",
    "pava_l1",
    "pava_l2",
    "pava_max",
)


def ordered_scores(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    items = sorted(
        ((int(key), float(value)) for key, value in raw.items()),
        key=lambda item: item[0],
    )
    if not items:
        raise ValueError("score JSON is empty")
    centers = np.asarray([item[0] for item in items], dtype=np.int64)
    scores = np.asarray([item[1] for item in items], dtype=np.float64)
    if np.any(np.diff(centers) <= 0):
        raise ValueError("score keys must be unique and numerically increasing")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite and in [0, 1]")
    return centers, scores


def repeat_interval_projection(
    scores: np.ndarray,
    frame_count: int,
    frame_interval: int,
    sigma: float,
) -> np.ndarray:
    """Reproduce the repository's existing repeat-N plus Gaussian evaluation."""
    smoothed = gaussian_filter1d(scores, sigma=sigma)
    projected = np.repeat(smoothed, frame_interval)
    if len(projected) < frame_count:
        projected = np.pad(projected, (0, frame_count - len(projected)))
    return projected[:frame_count]


def overlapping_window_projections(
    centers: np.ndarray,
    scores: np.ndarray,
    frame_count: int,
    fps: float,
    window_seconds: float,
) -> dict[str, np.ndarray]:
    """Project every clip to every frame it temporally covers."""
    if frame_count <= 0 or fps <= 0 or window_seconds <= 0:
        raise ValueError("frame_count, fps and window_seconds must be positive")
    sums = np.zeros(frame_count, dtype=np.float64)
    counts = np.zeros(frame_count, dtype=np.int32)
    maximum = np.full(frame_count, -np.inf, dtype=np.float64)
    top1 = np.full(frame_count, -np.inf, dtype=np.float64)
    top2 = np.full(frame_count, -np.inf, dtype=np.float64)
    top3 = np.full(frame_count, -np.inf, dtype=np.float64)
    half_window_frames = window_seconds * fps / 2.0

    for center, score in zip(centers, scores):
        start = max(0, int(math.ceil(center - half_window_frames - 1e-9)))
        stop = min(
            frame_count,
            int(math.floor(center + half_window_frames + 1e-9)) + 1,
        )
        if stop <= start:
            continue
        region = slice(start, stop)
        sums[region] += score
        counts[region] += 1
        np.maximum(maximum[region], score, out=maximum[region])

        first = top1[region].copy()
        second = top2[region].copy()
        third = top3[region].copy()
        insert_first = score >= first
        insert_second = (~insert_first) & (score >= second)
        insert_third = (~insert_first) & (~insert_second) & (score > third)
        third[insert_first] = second[insert_first]
        second[insert_first] = first[insert_first]
        first[insert_first] = score
        third[insert_second] = second[insert_second]
        second[insert_second] = score
        third[insert_third] = score
        top1[region], top2[region], top3[region] = first, second, third

    missing = counts == 0
    if np.any(missing):
        frame_indices = np.flatnonzero(missing)
        nearest = np.abs(frame_indices[:, None] - centers[None, :]).argmin(axis=1)
        fallback = scores[nearest]
        sums[missing] = fallback
        counts[missing] = 1
        maximum[missing] = fallback
        top1[missing] = fallback

    mean = sums / counts
    top_sum = top1.copy()
    at_least_two = counts >= 2
    at_least_three = counts >= 3
    top_sum[at_least_two] += top2[at_least_two]
    top_sum[at_least_three] += top3[at_least_three]
    top3_mean = top_sum / np.minimum(counts, 3)
    return {
        "overlap_mean": mean,
        "overlap_max": maximum,
        "overlap_top3": top3_mean,
    }


def normalized_binary_entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    entropy = -clipped * np.log(clipped) - (1.0 - clipped) * np.log1p(-clipped)
    return float(np.mean(entropy) / math.log(2.0))


def normalized_ordinal_entropy(monotonic_tail: np.ndarray) -> float:
    levels = np.concatenate(
        (
            [1.0 - monotonic_tail[0]],
            monotonic_tail[:-1] - monotonic_tail[1:],
            [monotonic_tail[-1]],
        )
    )
    levels = np.clip(levels, 0.0, 1.0)
    total = float(levels.sum())
    if total <= EPSILON:
        return 0.0
    levels /= total
    positive = levels > 0
    return float(-np.sum(levels[positive] * np.log(levels[positive])) / math.log(11.0))


def clip_uncertainty(detail: dict[str, Any]) -> dict[str, float]:
    threshold_rows = detail.get("thresholds", [])
    raw = np.asarray(
        [float(row["tail_probability"]) for row in threshold_rows],
        dtype=np.float64,
    )
    monotonic = np.asarray(
        detail.get("monotonic_tail_probabilities", []),
        dtype=np.float64,
    )
    if raw.shape != (10,) or monotonic.shape != (10,):
        raise ValueError("threshold detail must contain ten raw and ten PAVA probabilities")
    delta = raw - monotonic
    return {
        "tail_binary_entropy": normalized_binary_entropy(raw),
        "ordinal_entropy": normalized_ordinal_entropy(monotonic),
        "pava_l1": float(np.mean(np.abs(delta))),
        "pava_l2": float(np.sqrt(np.mean(np.square(delta)))),
        "pava_max": float(np.max(np.abs(delta))),
        "monotonic_violations": float(detail.get("monotonic_violations", 0)),
    }


def overlap_project_values(
    centers: np.ndarray,
    values: np.ndarray,
    frame_count: int,
    fps: float,
    window_seconds: float,
) -> np.ndarray:
    """Mean-project arbitrary clip diagnostics over their covered frames."""
    sums = np.zeros(frame_count, dtype=np.float64)
    counts = np.zeros(frame_count, dtype=np.int32)
    half_window_frames = window_seconds * fps / 2.0
    for center, value in zip(centers, values):
        start = max(0, int(math.ceil(center - half_window_frames - 1e-9)))
        stop = min(
            frame_count,
            int(math.floor(center + half_window_frames + 1e-9)) + 1,
        )
        sums[start:stop] += value
        counts[start:stop] += 1
    missing = counts == 0
    if np.any(missing):
        frame_indices = np.flatnonzero(missing)
        nearest = np.abs(frame_indices[:, None] - centers[None, :]).argmin(axis=1)
        sums[missing] = values[nearest]
        counts[missing] = 1
    return sums / counts


def metric_triplet(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, EPSILON)
    best_index = int(np.argmax(f1))
    if len(thresholds):
        threshold = float(thresholds[min(best_index, len(thresholds) - 1)])
    else:
        threshold = 0.5
    return {
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall, precision)),
        "max_f1": float(f1[best_index]),
        "max_f1_threshold": threshold,
    }


def group_summary(
    values: np.ndarray,
    masks: dict[str, np.ndarray],
    high_mask: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, mask in masks.items():
        selected = values[mask]
        result[name] = {
            "count": int(mask.sum()),
            "mean": float(np.mean(selected)) if len(selected) else float("nan"),
            "median": float(np.median(selected)) if len(selected) else float("nan"),
            "q75": float(np.quantile(selected, 0.75)) if len(selected) else float("nan"),
            "high_uncertainty_fraction": (
                float(np.mean(high_mask[mask])) if np.any(mask) else float("nan")
            ),
        }
    return result


def uncertainty_error_analysis(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    uncertainty: dict[str, np.ndarray],
    threshold: float,
    high_quantile: float,
) -> dict[str, Any]:
    predicted = baseline_scores >= threshold
    positive = labels.astype(bool)
    masks = {
        "TN": (~positive) & (~predicted),
        "FP": (~positive) & predicted,
        "FN": positive & (~predicted),
        "TP": positive & predicted,
    }
    error_mask = masks["FP"] | masks["FN"]
    output: dict[str, Any] = {
        "classification_threshold": threshold,
        "frame_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "overall_error_rate": float(np.mean(error_mask)),
        "measures": {},
    }
    for measure, values in uncertainty.items():
        cutoff = float(np.quantile(values, high_quantile))
        high = values >= cutoff
        high_error_rate = float(np.mean(error_mask[high])) if np.any(high) else float("nan")
        baseline_error_rate = float(np.mean(error_mask))
        try:
            error_auc = float(roc_auc_score(error_mask, values))
        except ValueError:
            error_auc = float("nan")
        predicted_positive = masks["FP"] | masks["TP"]
        predicted_negative = masks["FN"] | masks["TN"]
        try:
            fp_vs_tp_auc = float(
                roc_auc_score(masks["FP"][predicted_positive], values[predicted_positive])
            )
        except ValueError:
            fp_vs_tp_auc = float("nan")
        try:
            fn_vs_tn_auc = float(
                roc_auc_score(masks["FN"][predicted_negative], values[predicted_negative])
            )
        except ValueError:
            fn_vs_tn_auc = float("nan")
        correlation_result = spearmanr(values, baseline_scores)
        score_correlation = float(
            getattr(correlation_result, "statistic", correlation_result.correlation)
        )
        output["measures"][measure] = {
            "high_quantile": high_quantile,
            "high_cutoff": cutoff,
            "high_frame_fraction": float(np.mean(high)),
            "error_rate_among_high": high_error_rate,
            "error_enrichment": (
                high_error_rate / baseline_error_rate
                if baseline_error_rate > 0
                else float("nan")
            ),
            "error_detection_roc_auc": error_auc,
            "fp_vs_tp_roc_auc": fp_vs_tp_auc,
            "fn_vs_tn_roc_auc": fn_vs_tn_auc,
            "spearman_with_e0_score": score_correlation,
            "by_error_type": group_summary(values, masks, high),
        }
    return output


def write_metrics_csv(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "roc_auc", "pr_auc", "max_f1", "max_f1_threshold"),
        )
        writer.writeheader()
        for method in PROJECTION_NAMES:
            writer.writerow({"method": method, **metrics[method]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--annotation_file", type=Path, required=True)
    parser.add_argument("--temporal_annotation_file", type=Path, required=True)
    parser.add_argument("--scores_dir", type=Path, required=True)
    parser.add_argument("--threshold_details_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--normal_label", type=int, required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument("--smoothing_sigma", type=float, default=10.0)
    parser.add_argument("--high_uncertainty_quantile", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 < args.high_uncertainty_quantile < 1:
        parser.error("--high_uncertainty_quantile must lie in (0, 1)")
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
    all_labels: list[np.ndarray] = []
    all_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    all_uncertainty: dict[str, list[np.ndarray]] = defaultdict(list)
    processed = 0
    missing_details: list[str] = []

    for record in tqdm(records, desc="overlap projection", unit="video"):
        name = Path(record.path).name.removesuffix(".mp4")
        stem = output_stem(name)
        score_path = args.scores_dir / f"{stem}.json"
        raw_scores = load_json_dict(score_path)
        if not raw_scores:
            raise FileNotFoundError(f"missing score JSON: {score_path}")
        centers, scores = ordered_scores(raw_scores)
        video_path = resolve_video_path(args.video_dir, name)
        video_info = get_video_info(video_path)
        frame_count = int(record.num_frames)
        labels = np.asarray(
            get_video_labels(record, annotations, args.normal_label),
            dtype=np.int64,
        ) != args.normal_label
        projections = overlapping_window_projections(
            centers,
            scores,
            frame_count,
            video_info.fps,
            args.window_seconds,
        )
        projections["repeat16_sigma10"] = repeat_interval_projection(
            scores,
            frame_count,
            args.frame_interval,
            args.smoothing_sigma,
        )
        all_labels.append(labels)
        for method, values in projections.items():
            all_scores[method].append(values)

        details = load_json_dict(detail_dir / f"{stem}.json")
        if not details:
            missing_details.append(stem)
            continue
        diagnostics = {key: clip_uncertainty(details[key]) for key in details}
        if any(str(center) not in diagnostics for center in centers):
            missing_details.append(stem)
            continue
        per_video_output: dict[str, Any] = {}
        for measure in UNCERTAINTY_NAMES:
            clip_values = np.asarray(
                [diagnostics[str(center)][measure] for center in centers],
                dtype=np.float64,
            )
            frame_values = overlap_project_values(
                centers,
                clip_values,
                frame_count,
                video_info.fps,
                args.window_seconds,
            )
            all_uncertainty[measure].append(frame_values)
        for center in centers:
            per_video_output[str(center)] = diagnostics[str(center)]
        atomic_write_json(args.output_dir / "clip_uncertainty" / f"{stem}.json", per_video_output)
        processed += 1

    flat_labels = np.concatenate(all_labels)
    flat_scores = {
        method: np.concatenate(all_scores[method])
        for method in PROJECTION_NAMES
    }
    metrics = {
        method: metric_triplet(flat_labels, flat_scores[method])
        for method in PROJECTION_NAMES
    }
    write_metrics_csv(args.output_dir / "projection_metrics.csv", metrics)
    atomic_write_json(args.output_dir / "projection_metrics.json", metrics)

    summary: dict[str, Any] = {
        "projection_metrics": metrics,
        "videos_with_uncertainty": processed,
        "videos_missing_uncertainty": missing_details,
    }
    if processed == len(records):
        flat_uncertainty = {
            measure: np.concatenate(all_uncertainty[measure])
            for measure in UNCERTAINTY_NAMES
        }
        uncertainty_analysis = uncertainty_error_analysis(
            flat_labels,
            flat_scores["repeat16_sigma10"],
            flat_uncertainty,
            metrics["repeat16_sigma10"]["max_f1_threshold"],
            args.high_uncertainty_quantile,
        )
        summary["uncertainty_error_analysis"] = uncertainty_analysis
        atomic_write_json(
            args.output_dir / "uncertainty_error_analysis.json",
            uncertainty_analysis,
        )
    atomic_write_json(args.output_dir / "summary.json", summary)

    print("\nProjection metrics")
    print("method                     ROC-AUC    PR-AUC    Max-F1")
    for method in PROJECTION_NAMES:
        row = metrics[method]
        print(
            f"{method:25s} "
            f"{row['roc_auc'] * 100:8.4f}% "
            f"{row['pr_auc'] * 100:8.4f}% "
            f"{row['max_f1'] * 100:8.4f}%"
        )
    print(
        f"\nUncertainty details: {processed}/{len(records)} videos; "
        f"missing={len(missing_details)}"
    )
    if "uncertainty_error_analysis" in summary:
        analysis = summary["uncertainty_error_analysis"]
        print(
            f"Baseline frame error rate at Max-F1 threshold: "
            f"{analysis['overall_error_rate'] * 100:.4f}%"
        )
        for measure, row in analysis["measures"].items():
            print(
                f"{measure:24s} high-error="
                f"{row['error_rate_among_high'] * 100:.4f}% "
                f"enrichment={row['error_enrichment']:.4f} "
                f"error-AUC={row['error_detection_roc_auc']:.4f} "
                f"FP-vs-TP={row['fp_vs_tp_roc_auc']:.4f} "
                f"FN-vs-TN={row['fn_vs_tn_roc_auc']:.4f} "
                f"score-rho={row['spearman_with_e0_score']:.4f}"
            )
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
