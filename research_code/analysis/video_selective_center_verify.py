"""Selective 4-second E0 verification for near-boundary video clips.

Existing E0 score JSON files are ranked independently inside each video.  Only
clips in the configured percentile band are re-scored.  Verification uses the
unchanged E0 system prompt, ten YES/NO cumulative thresholds, shared visual
encoding, and PAVA.  No caption, tag, E3 calibration, refinement, or score
fusion is used.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.video_cumulative_score import CUMULATIVE_PROMPT
from src.video_score_utils import (
    CUMULATIVE_THRESHOLDS,
    DEFAULT_MODEL_PATH,
    VideoInfo,
    VideoWindow,
    atomic_write_json,
    build_threshold_conversation,
    cumulative_score_from_tail_probabilities,
    cumulative_threshold_likelihood_optimized,
    get_video_info,
    is_complete,
    load_json_dict,
    load_model,
    load_video_names,
    output_stem,
    resolve_video_path,
    split_for_job,
)

LOGGER = logging.getLogger("video_selective_center_verify")
EXPERIMENT = "e0_selective_center4_verify"


def ordered_initial_scores(raw: dict[str, Any]) -> list[tuple[int, float]]:
    ordered = sorted(
        ((int(key), float(value)) for key, value in raw.items()),
        key=lambda item: item[0],
    )
    if not ordered:
        raise ValueError("initial E0 score JSON is empty")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for _, score in ordered):
        raise ValueError("initial E0 scores must be finite and in [0, 1]")
    return ordered


def select_percentile_band(
    ordered: list[tuple[int, float]],
    lower_quantile: float,
    upper_quantile: float,
) -> list[dict[str, float | int]]:
    """Select an exact rank slice with deterministic center-frame tie breaking."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    count = len(ordered)
    start = min(count, int(math.floor(lower_quantile * count)))
    stop = min(count, int(math.ceil(upper_quantile * count)))
    if stop <= start:
        stop = min(count, start + 1)
    ranked = sorted(ordered, key=lambda item: (item[1], item[0]))
    selected_centers = {center for center, _ in ranked[start:stop]}
    rank_by_center = {
        center: rank
        for rank, (center, _) in enumerate(ranked)
    }
    denominator = max(1, count - 1)
    return [
        {
            "center_frame": center,
            "initial_score": score,
            "rank": rank_by_center[center],
            "percentile_rank": rank_by_center[center] / denominator,
        }
        for center, score in ordered
        if center in selected_centers
    ]


def centered_window(
    info: VideoInfo,
    center_frame: int,
    verify_seconds: float,
) -> VideoWindow:
    center_time = center_frame / info.fps
    return VideoWindow(
        center_frame=center_frame,
        start_time=max(0.0, center_time - verify_seconds / 2.0),
        end_time=min(info.duration, center_time + verify_seconds / 2.0),
    )


def process_video(args, model, processor, name: str) -> None:
    stem = output_stem(name)
    video_path = resolve_video_path(args.video_dir, name)
    initial_path = args.initial_scores_dir / f"{stem}.json"
    initial_raw = load_json_dict(initial_path)
    if not initial_raw:
        raise FileNotFoundError(f"missing initial E0 scores: {initial_path}")
    ordered = ordered_initial_scores(initial_raw)
    selected = select_percentile_band(
        ordered,
        args.lower_quantile,
        args.upper_quantile,
    )
    selection_by_key = {
        str(item["center_frame"]): item
        for item in selected
    }
    expected = list(selection_by_key)
    info = get_video_info(video_path)

    output_path = args.output_dir / f"{stem}.json"
    diagnostic_path = args.output_dir / "_verification_details" / f"{stem}.json"
    threshold_path = args.output_dir / "_threshold_details" / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    scores = load_json_dict(output_path) if args.resume else {}
    diagnostics = load_json_dict(diagnostic_path) if args.resume else {}
    threshold_details = (
        load_json_dict(threshold_path)
        if args.resume and args.save_threshold_details
        else {}
    )
    errors = load_json_dict(error_path) if args.resume else {}
    if (
        args.resume
        and is_complete(scores, expected)
        and is_complete(diagnostics, expected)
        and (
            not args.save_threshold_details
            or is_complete(threshold_details, expected)
        )
        and not errors
    ):
        LOGGER.info("Skipping complete video %s", stem)
        return

    dirty = 0

    def checkpoint() -> None:
        nonlocal dirty
        if not dirty:
            return
        atomic_write_json(output_path, scores)
        atomic_write_json(diagnostic_path, diagnostics)
        if args.save_threshold_details:
            atomic_write_json(threshold_path, threshold_details)
        if errors:
            atomic_write_json(error_path, errors)
        elif error_path.exists():
            error_path.unlink()
        dirty = 0

    LOGGER.info(
        "%s selected=%d/%d band=[%.3f, %.3f) verify=%.3fs",
        stem,
        len(selected),
        len(ordered),
        args.lower_quantile,
        args.upper_quantile,
        args.verify_seconds,
    )
    try:
        for item in tqdm(selected, desc=stem, unit="verify-clip"):
            key = str(item["center_frame"])
            if (
                args.resume
                and key in scores
                and key in diagnostics
                and (
                    not args.save_threshold_details
                    or key in threshold_details
                )
                and key not in errors
            ):
                continue
            window = centered_window(
                info,
                int(item["center_frame"]),
                args.verify_seconds,
            )
            try:
                conversations = [
                    build_threshold_conversation(
                        video_path,
                        window,
                        args.sample_fps,
                        args.max_frames,
                        args.prompt,
                        threshold,
                        precise_time=args.precise_time,
                    )
                    for threshold in CUMULATIVE_THRESHOLDS
                ]
                probabilities, raw_details = (
                    cumulative_threshold_likelihood_optimized(
                        model,
                        processor,
                        conversations,
                        yes_candidate=args.yes_candidate,
                        no_candidate=args.no_candidate,
                        temperature=args.likelihood_temperature,
                        threshold_batch_size=args.threshold_batch_size,
                        prefix_cache=args.prefix_cache,
                    )
                )
                verify_score, adjusted, violations = (
                    cumulative_score_from_tail_probabilities(
                        probabilities,
                        monotonic_projection=True,
                    )
                )
                initial_score = float(item["initial_score"])
                scores[key] = verify_score
                diagnostics[key] = {
                    "experiment": EXPERIMENT,
                    "center_frame": int(item["center_frame"]),
                    "initial_score": initial_score,
                    "verify_score": verify_score,
                    "delta": verify_score - initial_score,
                    "rank": int(item["rank"]),
                    "percentile_rank": float(item["percentile_rank"]),
                    "selection_band": [
                        args.lower_quantile,
                        args.upper_quantile,
                    ],
                    "verify_window": [window.start_time, window.end_time],
                    "verify_seconds_requested": args.verify_seconds,
                    "sample_fps": args.sample_fps,
                    "max_frames": args.max_frames,
                    "monotonic_violations": violations,
                }
                if args.save_threshold_details:
                    threshold_details[key] = {
                        "thresholds": [
                            {"threshold": threshold, **detail}
                            for threshold, detail in zip(
                                CUMULATIVE_THRESHOLDS,
                                raw_details,
                            )
                        ],
                        "monotonic_tail_probabilities": adjusted,
                        "monotonic_violations": violations,
                        "score": verify_score,
                    }
                errors.pop(key, None)
                LOGGER.info(
                    "%s frame=%s rank_pct=%.4f time=%.3f-%.3f "
                    "initial=%.6f verify=%.6f delta=%+.6f",
                    stem,
                    key,
                    float(item["percentile_rank"]),
                    window.start_time,
                    window.end_time,
                    initial_score,
                    verify_score,
                    verify_score - initial_score,
                )
            except Exception as exc:
                errors[key] = {
                    "error": repr(exc),
                    "center_frame": int(item["center_frame"]),
                    "initial_score": float(item["initial_score"]),
                    "verify_window": [window.start_time, window.end_time],
                }
                LOGGER.exception(
                    "%s frame=%s selective verification failed",
                    stem,
                    key,
                )
            dirty += 1
            if dirty >= args.checkpoint_interval:
                checkpoint()
    finally:
        checkpoint()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--index_file", type=Path)
    parser.add_argument("--video_list", nargs="+")
    parser.add_argument("--initial_scores_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lower_quantile", type=float, default=0.85)
    parser.add_argument("--upper_quantile", type=float, default=0.95)
    parser.add_argument("--verify_seconds", type=float, default=4.0)
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=2.5,
        help="2.5 FPS over four seconds yields ten uniformly sampled frames.",
    )
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument(
        "--precise_time",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--yes_candidate", default="YES")
    parser.add_argument("--no_candidate", default="NO")
    parser.add_argument("--likelihood_temperature", type=float, default=1.0)
    parser.add_argument("--threshold_batch_size", type=int, default=10)
    parser.add_argument(
        "--prefix_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument(
        "--save_threshold_details",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prompt", default=CUMULATIVE_PROMPT)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.lower_quantile < args.upper_quantile <= 1:
        parser.error("quantiles must satisfy 0 <= lower < upper <= 1")
    if args.verify_seconds <= 0 or args.sample_fps <= 0:
        parser.error("verify seconds and sample FPS must be positive")
    if args.max_frames <= 0:
        parser.error("--max_frames must be positive")
    if args.likelihood_temperature <= 0:
        parser.error("--likelihood_temperature must be positive")
    if args.threshold_batch_size <= 0 or args.checkpoint_interval <= 0:
        parser.error("batch and checkpoint intervals must be positive")
    if args.output_dir.resolve() == args.initial_scores_dir.resolve():
        parser.error("--output_dir must differ from --initial_scores_dir")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = split_for_job(
        load_video_names(args.video_dir, args.index_file, args.video_list),
        args.num_jobs,
        args.job_index,
    )
    model, processor = load_model(
        args.model_path,
        args.device,
        args.attn_implementation,
    )
    for name in names:
        try:
            process_video(args, model, processor, name)
        except Exception:
            LOGGER.exception("Failed to initialize/process video %s; continuing", name)


if __name__ == "__main__":
    main()
