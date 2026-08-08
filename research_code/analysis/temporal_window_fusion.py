"""Zero-training, score-level fusion across overlapping E0 video windows.

The input is an existing E0 score directory.  No VideoLLaMA3 inference,
caption, anomaly tag, refinement, label, or temporal annotation is used.
Output JSON files retain the exact ``{center_frame: score}`` interface.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from src.video_score_utils import (
    atomic_write_json,
    get_video_info,
    is_complete,
    load_json_dict,
    load_video_names,
    output_stem,
    resolve_video_path,
)

LOGGER = logging.getLogger("temporal_window_fusion")
SCORE_CLIP = 1e-4
EPSILON = 1e-6


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the deterministic lower weighted median."""
    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
        raise ValueError("values and weights must be equal-length 1D arrays")
    if len(values) == 0:
        raise ValueError("weighted median requires at least one value")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("weighted median inputs must be finite")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, total / 2.0, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, SCORE_CLIP, 1.0 - SCORE_CLIP)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exponential = math.exp(value)
    return float(exponential / (1.0 + exponential))


def fuse_score_sequence(
    center_frames: Sequence[int],
    scores: Sequence[float],
    fps: float,
    window_seconds: float = 10.0,
    mode: str = "adaptive",
) -> Tuple[np.ndarray, list[Dict[str, Any]]]:
    """Fuse one numerically ordered score sequence in physical time."""
    centers = np.asarray(center_frames, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if centers.ndim != 1 or values.ndim != 1 or len(centers) != len(values):
        raise ValueError("center_frames and scores must be equal-length 1D sequences")
    if len(values) == 0:
        return values.copy(), []
    if fps <= 0 or window_seconds <= 0:
        raise ValueError("fps and window_seconds must be positive")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("center frame keys must be strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("scores must lie in [0, 1]")
    if mode not in {
        "adjacent_mean",
        "overlap_mean",
        "overlap_logit",
        "adaptive",
    }:
        raise ValueError(f"unknown fusion mode: {mode}")

    times = centers.astype(np.float64) / fps
    radius = window_seconds / 2.0
    logits = _logit(values)
    fused = np.empty_like(values)
    diagnostics: list[Dict[str, Any]] = []

    for index, (time, score) in enumerate(zip(times, values)):
        if mode == "adjacent_mean":
            start = max(0, index - 1)
            stop = min(len(values), index + 2)
            neighbor_indices = np.arange(start, stop)
            neighbor_scores = values[neighbor_indices]
            result = float(np.mean(neighbor_scores))
            local_median = float(np.median(neighbor_scores))
            local_scale = None
            uncertainty = None
            fused[index] = min(1.0, max(0.0, result))
            diagnostics.append(
                {
                    "center_frame": int(centers[index]),
                    "center_time": float(time),
                    "neighbor_center_frames": [
                        int(centers[item]) for item in neighbor_indices
                    ],
                    "local_weighted_median": local_median,
                    "local_robust_scale": None,
                    "uncertainty_gate": None,
                    "initial_score": float(score),
                    "fused_score": float(fused[index]),
                    "delta": float(fused[index] - score),
                }
            )
            continue

        distances = np.abs(times - time)
        neighbor_indices = np.flatnonzero(distances <= radius + EPSILON)
        temporal_weights = np.maximum(
            0.0,
            1.0 - distances[neighbor_indices] / radius,
        )
        neighbor_scores = values[neighbor_indices]

        if mode == "overlap_mean":
            result = float(
                np.average(neighbor_scores, weights=temporal_weights)
            )
            local_median = weighted_median(neighbor_scores, temporal_weights)
            local_scale = None
            uncertainty = None
        elif mode == "overlap_logit":
            consensus_logit = float(
                np.average(logits[neighbor_indices], weights=temporal_weights)
            )
            result = _sigmoid(consensus_logit)
            local_median = weighted_median(neighbor_scores, temporal_weights)
            local_scale = None
            uncertainty = None
        else:
            local_median = weighted_median(
                neighbor_scores,
                temporal_weights,
            )
            local_mad = weighted_median(
                np.abs(neighbor_scores - local_median),
                temporal_weights,
            )
            local_scale = 1.4826 * local_mad
            score_distances = np.abs(neighbor_scores - score)
            if local_scale <= EPSILON:
                same_score = score_distances <= EPSILON
                same_score_support = float(temporal_weights[same_score].sum())
                total_support = float(temporal_weights.sum())
                if same_score_support >= total_support / 2.0:
                    # A sustained flat region or either side of a clean step:
                    # do not leak evidence across the boundary.
                    range_weights = same_score.astype(np.float64)
                else:
                    # An isolated center value has no temporal support. Allow
                    # the surrounding consensus to correct it.
                    range_weights = np.ones_like(score_distances)
            else:
                range_weights = np.exp(
                    -0.5 * np.square(score_distances / local_scale)
                )
            confidence_weights = (
                np.abs(2.0 * neighbor_scores - 1.0) + EPSILON
            )
            combined_weights = (
                temporal_weights * range_weights * confidence_weights
            )
            if float(combined_weights.sum()) <= EPSILON:
                consensus_logit = float(logits[index])
            else:
                consensus_logit = float(
                    np.average(
                        logits[neighbor_indices],
                        weights=combined_weights,
                    )
                )
            uncertainty = 4.0 * score * (1.0 - score)
            fused_logit = (
                (1.0 - uncertainty) * float(logits[index])
                + uncertainty * consensus_logit
            )
            result = _sigmoid(fused_logit)

        fused[index] = min(1.0, max(0.0, result))
        diagnostics.append(
            {
                "center_frame": int(centers[index]),
                "center_time": float(time),
                "neighbor_center_frames": [
                    int(centers[item]) for item in neighbor_indices
                ],
                "local_weighted_median": float(local_median),
                "local_robust_scale": (
                    None if local_scale is None else float(local_scale)
                ),
                "uncertainty_gate": (
                    None if uncertainty is None else float(uncertainty)
                ),
                "initial_score": float(score),
                "fused_score": float(fused[index]),
                "delta": float(fused[index] - score),
            }
        )
    return fused, diagnostics


def process_video(args, name: str) -> bool:
    stem = output_stem(name)
    input_path = args.scores_dir / f"{stem}.json"
    output_path = args.output_dir / f"{stem}.json"
    detail_path = args.output_dir / "_fusion_details" / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    raw_scores = load_json_dict(input_path)
    if not raw_scores:
        atomic_write_json(
            error_path,
            {"error": f"missing or empty input score JSON: {input_path}"},
        )
        LOGGER.error("%s: missing or empty input %s", stem, input_path)
        return False

    try:
        ordered = sorted(
            ((int(key), float(value)) for key, value in raw_scores.items()),
            key=lambda item: item[0],
        )
    except (TypeError, ValueError) as exc:
        atomic_write_json(error_path, {"error": repr(exc)})
        LOGGER.exception("%s: invalid score JSON", stem)
        return False
    expected = [str(center) for center, _ in ordered]
    existing = load_json_dict(output_path) if args.resume else {}
    existing_details = (
        load_json_dict(detail_path)
        if args.resume and args.save_details
        else {}
    )
    if (
        args.resume
        and is_complete(existing, expected)
        and (not args.save_details or set(existing_details) == set(expected))
        and not error_path.exists()
    ):
        return True

    try:
        video_path = resolve_video_path(args.video_dir, name)
        info = get_video_info(video_path)
        centers = [item[0] for item in ordered]
        initial = [item[1] for item in ordered]
        fused, diagnostics = fuse_score_sequence(
            centers,
            initial,
            info.fps,
            window_seconds=args.window_seconds,
            mode=args.fusion_mode,
        )
        output = {
            str(center): float(score)
            for center, score in zip(centers, fused)
        }
        atomic_write_json(output_path, output)
        if args.save_details:
            atomic_write_json(
                detail_path,
                {
                    str(center): detail
                    for center, detail in zip(centers, diagnostics)
                },
            )
        if error_path.exists():
            error_path.unlink()
        deltas = fused - np.asarray(initial, dtype=np.float64)
        LOGGER.info(
            "%s mode=%s fps=%.3f windows=%d changed=%d "
            "mean_abs_delta=%.6f max_abs_delta=%.6f",
            stem,
            args.fusion_mode,
            info.fps,
            len(fused),
            int(np.count_nonzero(np.abs(deltas) > 1e-12)),
            float(np.mean(np.abs(deltas))),
            float(np.max(np.abs(deltas))),
        )
        return True
    except Exception as exc:
        atomic_write_json(error_path, {"error": repr(exc)})
        LOGGER.exception("%s: fusion failed", stem)
        return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_dir", type=Path, required=True)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--index_file", type=Path)
    parser.add_argument("--video_list", nargs="+")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--fusion_mode",
        choices=[
            "adjacent_mean",
            "overlap_mean",
            "overlap_logit",
            "adaptive",
        ],
        default="adaptive",
    )
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument(
        "--save_details",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if args.window_seconds <= 0:
        parser.error("--window_seconds must be positive")
    if args.scores_dir.resolve() == args.output_dir.resolve():
        parser.error("--output_dir must differ from --scores_dir")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = load_video_names(args.video_dir, args.index_file, args.video_list)
    failures = 0
    for name in tqdm(names, desc=args.fusion_mode, unit="video"):
        failures += not process_video(args, name)
    if failures:
        raise SystemExit(f"{failures} video(s) failed; inspect {args.output_dir / '_errors'}")


if __name__ == "__main__":
    main()
