"""Analyze selective 4-second verification without combining final scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

from src.data.video_record import VideoRecord
from src.eval import get_video_labels, temporal_testing_annotations
from src.eval_overlap_projection_uncertainty import (
    metric_triplet,
    ordered_scores,
    repeat_interval_projection,
)
from src.video_score_utils import atomic_write_json, load_json_dict, output_stem


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def delta_summary(delta: np.ndarray) -> dict[str, float | int]:
    if not len(delta):
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "raised_fraction": float("nan"),
            "lowered_fraction": float("nan"),
        }
    return {
        "count": int(len(delta)),
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
        "q25": float(np.quantile(delta, 0.25)),
        "q75": float(np.quantile(delta, 0.75)),
        "raised_fraction": float(np.mean(delta > 0)),
        "lowered_fraction": float(np.mean(delta < 0)),
    }


def analyze_scope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["is_anomalous"] for row in rows], dtype=bool)
    initial = np.asarray([row["initial_score"] for row in rows], dtype=np.float64)
    verify = np.asarray([row["verify_score"] for row in rows], dtype=np.float64)
    delta = verify - initial
    fn_delta = delta[labels]
    tn_delta = delta[~labels]
    return {
        "clips": int(len(rows)),
        "positive_clips": int(labels.sum()),
        "negative_clips": int((~labels).sum()),
        "initial_fn_tn_roc_auc": safe_auc(labels, initial),
        "verify_fn_tn_roc_auc": safe_auc(labels, verify),
        "auc_change": safe_auc(labels, verify) - safe_auc(labels, initial),
        "delta_all": delta_summary(delta),
        "delta_fn": delta_summary(fn_delta),
        "delta_tn": delta_summary(tn_delta),
        "mean_delta_gap_fn_minus_tn": (
            float(np.mean(fn_delta) - np.mean(tn_delta))
            if len(fn_delta) and len(tn_delta)
            else float("nan")
        ),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "video",
        "center_frame",
        "percentile_rank",
        "initial_score",
        "smoothed_initial_score",
        "verify_score",
        "delta",
        "anomaly_fraction",
        "is_anomalous",
        "analysis_predicted_normal",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--annotation_file", type=Path, required=True)
    parser.add_argument("--temporal_annotation_file", type=Path, required=True)
    parser.add_argument("--initial_scores_dir", type=Path, required=True)
    parser.add_argument("--verify_scores_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--normal_label", type=int, required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--smoothing_sigma", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = temporal_testing_annotations(args.temporal_annotation_file)
    records = [
        VideoRecord(line.strip().split(), args.root_path)
        for line in args.annotation_file.read_text().splitlines()
        if line.strip()
    ]

    full_frame_labels: list[np.ndarray] = []
    full_frame_scores: list[np.ndarray] = []
    cached: list[dict[str, Any]] = []
    for record in records:
        name = Path(record.path).name.removesuffix(".mp4")
        stem = output_stem(name)
        initial_raw = load_json_dict(args.initial_scores_dir / f"{stem}.json")
        verify_raw = load_json_dict(args.verify_scores_dir / f"{stem}.json")
        details = load_json_dict(
            args.verify_scores_dir / "_verification_details" / f"{stem}.json"
        )
        errors = load_json_dict(args.verify_scores_dir / "_errors" / f"{stem}.json")
        if not initial_raw:
            raise FileNotFoundError(f"missing initial E0 scores for {stem}")
        if errors:
            raise RuntimeError(f"verification errors remain for {stem}: {len(errors)}")
        if not verify_raw or set(verify_raw) != set(details):
            raise RuntimeError(f"incomplete verification output for {stem}")
        centers, initial = ordered_scores(initial_raw)
        smoothed = gaussian_filter1d(initial, sigma=args.smoothing_sigma)
        center_to_index = {
            str(center): index
            for index, center in enumerate(centers)
        }
        labels = np.asarray(
            get_video_labels(record, annotations, args.normal_label),
            dtype=np.int64,
        ) != args.normal_label
        full_frame_labels.append(labels)
        full_frame_scores.append(
            repeat_interval_projection(
                initial,
                len(labels),
                args.frame_interval,
                args.smoothing_sigma,
            )
        )
        cached.append(
            {
                "stem": stem,
                "labels": labels,
                "initial_raw": initial_raw,
                "smoothed": smoothed,
                "center_to_index": center_to_index,
                "verify_raw": verify_raw,
                "details": details,
            }
        )

    baseline = metric_triplet(
        np.concatenate(full_frame_labels),
        np.concatenate(full_frame_scores),
    )
    analysis_threshold = baseline["max_f1_threshold"]
    rows: list[dict[str, Any]] = []
    for video in cached:
        for key, verify_value in video["verify_raw"].items():
            index = video["center_to_index"].get(key)
            if index is None:
                raise KeyError(f"{video['stem']} verify key {key} absent from E0")
            center = int(key)
            start = center
            stop = min(len(video["labels"]), center + args.frame_interval)
            if stop <= start:
                continue
            anomaly_fraction = float(np.mean(video["labels"][start:stop]))
            initial_score = float(video["initial_raw"][key])
            verify_score = float(verify_value)
            smoothed_score = float(video["smoothed"][index])
            detail = video["details"][key]
            rows.append(
                {
                    "video": video["stem"],
                    "center_frame": center,
                    "percentile_rank": float(detail["percentile_rank"]),
                    "initial_score": initial_score,
                    "smoothed_initial_score": smoothed_score,
                    "verify_score": verify_score,
                    "delta": verify_score - initial_score,
                    "anomaly_fraction": anomaly_fraction,
                    "is_anomalous": anomaly_fraction >= 0.5,
                    "analysis_predicted_normal": smoothed_score < analysis_threshold,
                }
            )

    all_selected = analyze_scope(rows)
    predicted_normal_rows = [
        row for row in rows if row["analysis_predicted_normal"]
    ]
    predicted_normal = analyze_scope(predicted_normal_rows)
    summary = {
        "no_score_fusion_performed": True,
        "baseline_frame_metrics": baseline,
        "analysis_only_max_f1_threshold": analysis_threshold,
        "clip_ground_truth": (
            "anomalous when at least 50% of the corresponding 16-frame "
            "evaluation block is anomalous"
        ),
        "all_selected_85_95": all_selected,
        "analysis_predicted_normal_fn_tn": predicted_normal,
    }
    atomic_write_json(args.output_dir / "selective_verify_analysis.json", summary)
    write_rows(args.output_dir / "selective_verify_clips.csv", rows)

    print("\nSelective center-4s verification (no fusion)")
    print(f"Analysis-only threshold: {analysis_threshold:.9f}")
    for name, result in (
        ("all selected clips", all_selected),
        ("predicted-normal FN/TN", predicted_normal),
    ):
        print(f"\n{name}: clips={result['clips']}")
        print(
            f"initial AUC={result['initial_fn_tn_roc_auc'] * 100:.4f}% "
            f"verify AUC={result['verify_fn_tn_roc_auc'] * 100:.4f}% "
            f"change={result['auc_change'] * 100:+.4f} pp"
        )
        print(
            f"FN mean delta={result['delta_fn']['mean']:+.6f} "
            f"TN mean delta={result['delta_tn']['mean']:+.6f} "
            f"gap={result['mean_delta_gap_fn_minus_tn']:+.6f}"
        )
        print(
            f"FN raised={result['delta_fn']['raised_fraction'] * 100:.2f}% "
            f"TN raised={result['delta_tn']['raised_fraction'] * 100:.2f}%"
        )
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
