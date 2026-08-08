"""Recompute a five-threshold cumulative score from saved E0 details.

This is a post-hoc ablation: it never loads VideoLLaMA3.  For every clip it
selects the raw tail probabilities at the requested thresholds, applies the
same decreasing PAVA projection used by E0, and averages the projected values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.video_score_utils import decreasing_isotonic_projection


def _load_dict(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _threshold_value(entries: list[dict], target: float) -> float:
    candidates = [
        item
        for item in entries
        if isinstance(item, dict) and "threshold" in item
    ]
    if not candidates:
        raise ValueError("threshold list is empty")
    item = min(candidates, key=lambda row: abs(float(row["threshold"]) - target))
    if abs(float(item["threshold"]) - target) > 1e-6:
        raise ValueError(f"threshold {target} is missing")
    return float(item["tail_probability"])


def convert_file(
    score_path: Path,
    detail_path: Path,
    output_path: Path,
    thresholds: list[float],
) -> tuple[int, int]:
    scores = _load_dict(score_path)
    details = _load_dict(detail_path)
    output: dict[str, float] = {}
    missing = 0
    for center, _ in scores.items():
        detail = details.get(center)
        if not isinstance(detail, dict):
            missing += 1
            continue
        try:
            raw = [
                _threshold_value(detail["thresholds"], threshold)
                for threshold in thresholds
            ]
            projected = decreasing_isotonic_projection(raw)
            output[str(center)] = float(sum(projected) / len(projected))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            missing += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return len(output), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_dir", type=Path, required=True)
    parser.add_argument("--threshold_details_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default="0.2,0.4,0.6,0.8,1.0",
        help="comma-separated cumulative thresholds",
    )
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",") if value]
    if not thresholds:
        raise ValueError("at least one threshold is required")

    details_dir = args.threshold_details_dir or args.scores_dir / "_threshold_details"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_videos = total_clips = missing_clips = 0
    for score_path in sorted(args.scores_dir.glob("*.json")):
        detail_path = details_dir / score_path.name
        if not detail_path.is_file():
            print(f"missing detail file: {detail_path}")
            continue
        count, missing = convert_file(
            score_path,
            detail_path,
            args.output_dir / score_path.name,
            thresholds,
        )
        total_videos += 1
        total_clips += count
        missing_clips += missing

    metadata = {
        "source_scores_dir": str(args.scores_dir),
        "threshold_details_dir": str(details_dir),
        "thresholds": thresholds,
        "videos_converted": total_videos,
        "clips_converted": total_clips,
        "clips_missing_details": missing_clips,
        "pava": "decreasing_isotonic_projection",
        "score": "mean(projected_tail_probabilities)",
    }
    (args.output_dir / "threshold5_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
