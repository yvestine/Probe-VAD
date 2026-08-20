"""CPU-only sensitivity analysis for already generated E0 score JSON files.

This script never loads VideoLLaMA3.  It evaluates Gaussian smoothing values,
optionally compares raw-tail aggregation with PAVA aggregation when threshold
details are available, and estimates video-bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.data.video_record import VideoRecord
from src.eval_interpolated import get_video_labels, temporal_testing_annotations


def metric_triplet(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return {
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall, precision)),
        "max_f1": float(f1[best]),
        "max_f1_threshold": float(thresholds[min(best, len(thresholds) - 1)])
        if len(thresholds)
        else 0.5,
    }


def load_score(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text())
    return {str(k): float(v) for k, v in value.items() if str(k).lstrip("-").isdigit()}


def score_from_details(detail_path: Path, pava: bool) -> dict[str, float]:
    details = json.loads(detail_path.read_text())
    output: dict[str, float] = {}
    for center, row in details.items():
        if pava:
            values = row.get("monotonic_tail_probabilities", [])
        else:
            values = [item["tail_probability"] for item in row.get("thresholds", [])]
        if len(values) == 10:
            output[str(center)] = float(np.mean(values))
    return output


def interpolate(score: dict[str, float], frame_count: int, sigma: float) -> np.ndarray:
    ordered = sorted(score.items(), key=lambda item: int(item[0]))
    centers = np.asarray([int(k) for k, _ in ordered], dtype=np.float64)
    values = np.asarray([v for _, v in ordered], dtype=np.float64)
    output_centers = np.arange(0, frame_count, 16, dtype=np.float64)
    values = np.interp(output_centers, centers, values)
    if sigma > 0 and len(values) > 1:
        values = gaussian_filter1d(values, sigma=sigma)
    return np.repeat(values, 16)[:frame_count]


def load_videos(index_path: Path, root_path: str) -> list[VideoRecord]:
    return [
        VideoRecord(line.strip().split(), root_path)
        for line in index_path.read_text().splitlines()
        if line.strip()
    ]


def collect_video_arrays(
    videos: list[VideoRecord],
    scores_dir: Path,
    details_dir: Path | None,
    annotations: dict[str, list[str]],
    normal_label: int,
    sigma: float,
    pava: bool | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    names: list[str] = []
    for video in videos:
        name = Path(video.path).name.removesuffix(".mp4")
        score_path = scores_dir / f"{name}.json"
        if not score_path.is_file():
            continue
        score = load_score(score_path)
        if details_dir is not None and pava is not None:
            detail_path = details_dir / f"{name}.json"
            if not detail_path.is_file():
                continue
            score = score_from_details(detail_path, pava)
        if not score:
            continue
        scores = interpolate(score, video.num_frames, sigma)
        labels = np.asarray(get_video_labels(video, annotations, normal_label)) != normal_label
        n = min(len(scores), len(labels))
        all_scores.append(scores[:n])
        all_labels.append(labels[:n])
        names.append(name)
    return all_scores, all_labels, names


def pooled_metrics(score_arrays: list[np.ndarray], label_arrays: list[np.ndarray]) -> dict[str, float]:
    if not score_arrays:
        raise RuntimeError("no score files matched the index")
    return metric_triplet(np.concatenate(label_arrays), np.concatenate(score_arrays))


def bootstrap(
    score_arrays: list[np.ndarray],
    label_arrays: list[np.ndarray],
    reps: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(score_arrays)
    values = {key: [] for key in ("roc_auc", "pr_auc", "max_f1")}
    for _ in range(reps):
        selected = rng.integers(0, n, size=n)
        scores = np.concatenate([score_arrays[i] for i in selected])
        labels = np.concatenate([label_arrays[i] for i in selected])
        metrics = metric_triplet(labels, scores)
        for key in values:
            values[key].append(metrics[key])
    return {
        key: {
            "mean": float(np.mean(vals)),
            "lower_2.5": float(np.quantile(vals, 0.025)),
            "upper_97.5": float(np.quantile(vals, 0.975)),
        }
        for key, vals in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_dir", type=Path, required=True)
    parser.add_argument("--index_file", type=Path, required=True)
    parser.add_argument("--temporal_annotation_file", type=Path, required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--normal_label", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sigmas", default="0,2,5,10,20")
    parser.add_argument("--bootstrap_reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threshold_details_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = load_videos(args.index_file, args.root_path)
    annotations = temporal_testing_annotations(args.temporal_annotation_file)

    sigma_rows: list[dict[str, float | int]] = []
    for sigma_text in args.sigmas.split(","):
        sigma = float(sigma_text)
        scores, labels, names = collect_video_arrays(
            videos, args.scores_dir, None, annotations, args.normal_label, sigma
        )
        metrics = pooled_metrics(scores, labels)
        sigma_rows.append({"sigma": sigma, "videos": len(names), **metrics})
    with (args.output_dir / "sigma_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sigma_rows[0].keys())
        writer.writeheader()
        writer.writerows(sigma_rows)
    (args.output_dir / "sigma_sweep.json").write_text(json.dumps(sigma_rows, indent=2))

    sigma10_scores, sigma10_labels, names = collect_video_arrays(
        videos, args.scores_dir, None, annotations, args.normal_label, 10.0
    )
    bootstrap_result = {
        "videos": len(names),
        "repetitions": args.bootstrap_reps,
        "seed": args.seed,
        "metrics": pooled_metrics(sigma10_scores, sigma10_labels),
        "confidence_intervals": bootstrap(
            sigma10_scores, sigma10_labels, args.bootstrap_reps, args.seed
        ),
    }
    (args.output_dir / "bootstrap_sigma10.json").write_text(
        json.dumps(bootstrap_result, indent=2)
    )

    if args.threshold_details_dir is not None:
        pava_rows = []
        for pava in (False, True):
            scores, labels, names = collect_video_arrays(
                videos,
                args.scores_dir,
                args.threshold_details_dir,
                annotations,
                args.normal_label,
                10.0,
                pava,
            )
            pava_rows.append({"pava": pava, "videos": len(names), **pooled_metrics(scores, labels)})
        (args.output_dir / "pava_on_off.json").write_text(json.dumps(pava_rows, indent=2))


if __name__ == "__main__":
    main()
