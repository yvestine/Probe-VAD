"""E4 direct scoring with center-dense temporal frame sampling.

E4 changes only the temporal placement of the ten visual frames used by E0:
four frames cover the complete sliding window and six frames densely cover a
four-second interval around its center.  The E0 system prompt, ten YES/NO
threshold questions, shared visual encoding, PAVA projection, score JSON
format, and evaluation interface remain unchanged.  No caption, anomaly tag,
refinement, or score fusion is used.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm

from src.video_cumulative_score import CUMULATIVE_PROMPT
from src.video_score_utils import (
    CUMULATIVE_THRESHOLDS,
    DEFAULT_MODEL_PATH,
    VideoWindow,
    atomic_write_json,
    build_threshold_conversation,
    cumulative_score_from_tail_probabilities,
    cumulative_threshold_likelihood_optimized,
    get_video_info,
    is_complete,
    iter_video_windows,
    load_json_dict,
    load_model,
    load_video_names,
    output_stem,
    resolve_video_path,
    split_for_job,
    threshold_yes_no_likelihood,
)

LOGGER = logging.getLogger("video_center_dense_score")
EXPERIMENT = "e4_center_dense"


def _uniform_targets(start: float, end: float, count: int) -> List[float]:
    """Return deterministic inclusive targets without requiring NumPy."""
    if count <= 0:
        return []
    if count == 1 or end <= start:
        return [(start + end) / 2.0]
    step = (end - start) / (count - 1)
    return [start + index * step for index in range(count)]


def center_dense_interval(
    window_start: float,
    window_end: float,
    center_time: float,
    center_seconds: float,
) -> Tuple[float, float]:
    """Place a fixed center interval inside a possibly boundary-clipped window.

    Near the beginning or end of a video the interval is shifted, rather than
    shortened, whenever the available window is at least ``center_seconds``.
    """
    if window_end < window_start:
        raise ValueError("window_end must not be before window_start")
    if center_seconds <= 0:
        raise ValueError("center_seconds must be positive")
    duration = window_end - window_start
    span = min(center_seconds, duration)
    if duration <= span:
        return window_start, window_end
    start = min(
        max(center_time - span / 2.0, window_start),
        window_end - span,
    )
    return start, start + span


def _take_nearest(
    timestamps: Sequence[float],
    targets: Sequence[float],
    available: set[int],
    allowed: set[int] | None = None,
) -> List[int]:
    selected: List[int] = []
    for target in targets:
        candidates = available if allowed is None else available.intersection(allowed)
        if not candidates:
            break
        index = min(
            candidates,
            key=lambda candidate: (
                abs(float(timestamps[candidate]) - target),
                candidate,
            ),
        )
        selected.append(index)
        available.remove(index)
    return selected


def select_center_dense_indices(
    timestamps: Sequence[float],
    window_start: float,
    window_end: float,
    center_time: float,
    global_frames: int = 4,
    center_frames: int = 6,
    center_seconds: float = 4.0,
) -> Dict[str, Any]:
    """Select four global and six center-dense candidate frames.

    Unique frames are used whenever at least ``global_frames + center_frames``
    candidates exist.  For very short clips all available frames are retained
    first, then the frame nearest the window center is repeated deterministically
    until the requested count is reached.
    """
    if not timestamps:
        raise ValueError("video decoder returned no candidate frames")
    if global_frames < 0 or center_frames < 0:
        raise ValueError("frame counts must be non-negative")
    requested = global_frames + center_frames
    if requested <= 0:
        raise ValueError("at least one frame must be requested")

    numeric_timestamps = [float(value) for value in timestamps]
    center_start, center_end = center_dense_interval(
        window_start,
        window_end,
        center_time,
        center_seconds,
    )
    available = set(range(len(numeric_timestamps)))

    global_targets = _uniform_targets(
        numeric_timestamps[0],
        numeric_timestamps[-1],
        global_frames,
    )
    global_indices = _take_nearest(
        numeric_timestamps,
        global_targets,
        available,
    )

    center_targets = _uniform_targets(center_start, center_end, center_frames)
    center_allowed = {
        index
        for index, timestamp in enumerate(numeric_timestamps)
        if center_start - 1e-9 <= timestamp <= center_end + 1e-9
    }
    center_indices = _take_nearest(
        numeric_timestamps,
        center_targets,
        available,
        allowed=center_allowed,
    )

    # A coarse candidate grid or a global frame inside the center band may
    # leave fewer than six unused in-band frames. Prefer candidates nearest the
    # corresponding dense targets, then any unused candidates nearest center.
    missing_center = center_frames - len(center_indices)
    if missing_center > 0:
        remaining_targets = center_targets[len(center_indices):]
        center_indices.extend(
            _take_nearest(
                numeric_timestamps,
                remaining_targets,
                available,
            )
        )
    missing_center = center_frames - len(center_indices)
    if missing_center > 0 and available:
        center_indices.extend(
            _take_nearest(
                numeric_timestamps,
                [center_time] * missing_center,
                available,
            )
        )

    selected = global_indices + center_indices
    if len(selected) < requested and available:
        selected.extend(
            _take_nearest(
                numeric_timestamps,
                [center_time] * (requested - len(selected)),
                available,
            )
        )
    if len(selected) < requested:
        # This path is used only when the decoder returned fewer candidates
        # than requested. Repeating a real frame is preferable to failing the
        # whole boundary window or silently changing the visual token count.
        pad_index = min(
            range(len(numeric_timestamps)),
            key=lambda index: (
                abs(numeric_timestamps[index] - center_time),
                index,
            ),
        )
        selected.extend([pad_index] * (requested - len(selected)))

    selected = sorted(
        selected,
        key=lambda index: (numeric_timestamps[index], index),
    )
    return {
        "indices": selected,
        "global_indices": global_indices,
        "center_indices": center_indices,
        "center_interval": [center_start, center_end],
    }


def load_center_dense_clip(
    processor,
    video_path: Path,
    window: VideoWindow,
    center_time: float,
    sample_fps: float,
    max_frames: int,
    global_frames: int,
    center_frames: int,
    center_seconds: float,
    decode_max_frames: int,
    precise_time: bool,
) -> Tuple[List[Any], List[float], Dict[str, Any]]:
    """Decode the 2-FPS candidate pool once and select E4's ten frames."""
    expected_candidates = math.ceil(
        max(0.0, window.end_time - window.start_time) * sample_fps
    ) + 2
    candidate_limit = max(decode_max_frames, expected_candidates, max_frames)
    frames, timestamps = processor.load_video(
        video_path=str(video_path),
        start_time=window.start_time,
        end_time=window.end_time,
        fps=sample_fps,
        max_frames=candidate_limit,
        precise_time=precise_time,
    )
    if len(frames) != len(timestamps):
        raise RuntimeError(
            "decoder returned mismatched frames/timestamps: "
            f"{len(frames)} != {len(timestamps)}"
        )
    selection = select_center_dense_indices(
        timestamps,
        window.start_time,
        window.end_time,
        center_time,
        global_frames=global_frames,
        center_frames=center_frames,
        center_seconds=center_seconds,
    )
    indices = selection["indices"]
    selected_frames = [frames[index] for index in indices]
    selected_timestamps = [float(timestamps[index]) for index in indices]
    selection.update(
        {
            "candidate_count": len(frames),
            "candidate_timestamps": [float(value) for value in timestamps],
            "selected_timestamps": selected_timestamps,
            "global_timestamps": [
                float(timestamps[index])
                for index in selection["global_indices"]
            ],
            "center_timestamps": [
                float(timestamps[index])
                for index in selection["center_indices"]
            ],
        }
    )
    return selected_frames, selected_timestamps, selection


def build_preloaded_threshold_conversation(
    video_path: Path,
    window: VideoWindow,
    sample_fps: float,
    max_frames: int,
    prompt: str,
    threshold: float,
    frames: Sequence[Any],
    timestamps: Sequence[float],
    precise_time: bool = False,
) -> List[Dict[str, Any]]:
    """Reuse the exact E0 prompt while replacing its decoder arguments."""
    conversation = build_threshold_conversation(
        video_path,
        window,
        sample_fps,
        max_frames,
        prompt,
        threshold,
        precise_time=precise_time,
    )
    video_item = conversation[1]["content"][0]
    video_item["video"] = list(frames)
    video_item["num_frames"] = len(frames)
    video_item["timestamps"] = [float(value) for value in timestamps]
    return conversation


def process_video(args, model, processor, name: str) -> None:
    video_path = resolve_video_path(args.video_dir, name)
    stem = output_stem(name)
    output_path = args.output_dir / f"{stem}.json"
    threshold_path = args.output_dir / "_threshold_details" / f"{stem}.json"
    sampling_path = args.output_dir / "_sampling_details" / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    info = get_video_info(video_path)
    windows = list(iter_video_windows(info, args.frame_interval, args.window_seconds))
    expected = [str(window.center_frame) for window in windows]
    scores = load_json_dict(output_path) if args.resume else {}
    threshold_details = (
        load_json_dict(threshold_path)
        if args.resume and args.save_threshold_details
        else {}
    )
    sampling_details = (
        load_json_dict(sampling_path)
        if args.resume and args.save_sampling_details
        else {}
    )
    errors = load_json_dict(error_path) if args.resume else {}
    if (
        args.resume
        and is_complete(scores, expected)
        and (
            not args.save_threshold_details
            or set(threshold_details) == set(expected)
        )
        and (
            not args.save_sampling_details
            or set(sampling_details) == set(expected)
        )
        and not errors
    ):
        LOGGER.info("Skipping complete video %s", stem)
        return

    LOGGER.info(
        "Video %s experiment=%s: %d frames, %.3f FPS, %d windows",
        stem,
        EXPERIMENT,
        info.frame_count,
        info.fps,
        len(windows),
    )
    dirty_windows = 0

    def checkpoint() -> None:
        nonlocal dirty_windows
        if dirty_windows == 0:
            return
        atomic_write_json(output_path, scores)
        if args.save_threshold_details:
            atomic_write_json(threshold_path, threshold_details)
        if args.save_sampling_details:
            atomic_write_json(sampling_path, sampling_details)
        if errors:
            atomic_write_json(error_path, errors)
        elif error_path.exists():
            error_path.unlink()
        dirty_windows = 0

    try:
        for window in tqdm(windows, desc=stem, unit="window"):
            key = str(window.center_frame)
            if (
                args.resume
                and key in scores
                and (
                    not args.save_threshold_details
                    or key in threshold_details
                )
                and (
                    not args.save_sampling_details
                    or key in sampling_details
                )
                and key not in errors
            ):
                continue
            try:
                center_time = window.center_frame / info.fps
                frames, timestamps, selection = load_center_dense_clip(
                    processor,
                    video_path,
                    window,
                    center_time,
                    args.sample_fps,
                    args.max_frames,
                    args.global_frames,
                    args.center_frames,
                    args.center_seconds,
                    args.decode_max_frames,
                    args.precise_time,
                )
                conversations = [
                    build_preloaded_threshold_conversation(
                        video_path,
                        window,
                        args.sample_fps,
                        args.max_frames,
                        args.prompt,
                        threshold,
                        frames,
                        timestamps,
                        precise_time=args.precise_time,
                    )
                    for threshold in CUMULATIVE_THRESHOLDS
                ]
                probabilities: List[float] = []
                raw_threshold_details: List[Dict[str, Any]] = []
                if args.optimized:
                    probabilities, optimized_details = (
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
                    if args.save_threshold_details:
                        raw_threshold_details = [
                            {"threshold": threshold, **detail}
                            for threshold, detail in zip(
                                CUMULATIVE_THRESHOLDS,
                                optimized_details,
                            )
                        ]
                else:
                    for threshold, conversation in zip(
                        CUMULATIVE_THRESHOLDS,
                        conversations,
                    ):
                        probability, yes_ll, no_ll = threshold_yes_no_likelihood(
                            model,
                            processor,
                            conversation,
                            yes_candidate=args.yes_candidate,
                            no_candidate=args.no_candidate,
                            length_normalize=args.length_normalize,
                            temperature=args.likelihood_temperature,
                        )
                        probabilities.append(probability)
                        if args.save_threshold_details:
                            raw_threshold_details.append(
                                {
                                    "threshold": threshold,
                                    "yes_loglik": yes_ll,
                                    "no_loglik": no_ll,
                                    "tail_probability": probability,
                                }
                            )
                score, adjusted, violations = (
                    cumulative_score_from_tail_probabilities(
                        probabilities,
                        monotonic_projection=args.monotonic_projection,
                    )
                )
                scores[key] = score
                if args.save_threshold_details:
                    threshold_details[key] = {
                        "experiment": EXPERIMENT,
                        "thresholds": raw_threshold_details,
                        "monotonic_tail_probabilities": adjusted,
                        "monotonic_violations": violations,
                        "score": score,
                    }
                if args.save_sampling_details:
                    sampling_details[key] = {
                        "experiment": EXPERIMENT,
                        "window": [window.start_time, window.end_time],
                        "center_time": center_time,
                        **selection,
                    }
                errors.pop(key, None)
                LOGGER.info(
                    "%s experiment=%s frame=%s time=%.3f-%.3f "
                    "sampled=%s score=%.6f tail_raw=%s "
                    "tail_monotonic=%s violations=%d",
                    stem,
                    EXPERIMENT,
                    key,
                    window.start_time,
                    window.end_time,
                    [round(value, 3) for value in timestamps],
                    score,
                    [round(value, 6) for value in probabilities],
                    [round(value, 6) for value in adjusted],
                    violations,
                )
            except Exception as exc:
                scores[key] = args.default_score
                errors[key] = {"raw_output": "", "error": repr(exc)}
                LOGGER.exception(
                    "%s experiment=%s frame=%s time=%.3f-%.3f failed; "
                    "using default %.3f",
                    stem,
                    EXPERIMENT,
                    key,
                    window.start_time,
                    window.end_time,
                    args.default_score,
                )
            dirty_windows += 1
            if dirty_windows >= args.checkpoint_interval:
                checkpoint()
    finally:
        checkpoint()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--index_file", type=Path)
    parser.add_argument("--video_list", nargs="+")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--global_frames", type=int, default=4)
    parser.add_argument("--center_frames", type=int, default=6)
    parser.add_argument("--center_seconds", type=float, default=4.0)
    parser.add_argument(
        "--decode_max_frames",
        type=int,
        default=64,
        help="Safety cap for the pre-selection 2-FPS candidate pool.",
    )
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
    parser.add_argument(
        "--length_normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--likelihood_temperature", type=float, default=1.0)
    parser.add_argument(
        "--optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--threshold_batch_size", type=int, default=10)
    parser.add_argument(
        "--prefix_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--monotonic_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--default_score", type=float, default=0.5)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument(
        "--save_threshold_details",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save_sampling_details",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prompt", default=CUMULATIVE_PROMPT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if args.frame_interval <= 0:
        parser.error("--frame_interval must be positive")
    if args.window_seconds <= 0:
        parser.error("--window_seconds must be positive")
    if args.sample_fps <= 0:
        parser.error("--sample_fps must be positive")
    if args.global_frames < 0 or args.center_frames < 0:
        parser.error("--global_frames and --center_frames must be non-negative")
    if args.global_frames + args.center_frames != args.max_frames:
        parser.error(
            "--global_frames + --center_frames must equal --max_frames"
        )
    if args.center_seconds <= 0:
        parser.error("--center_seconds must be positive")
    if args.decode_max_frames < args.max_frames:
        parser.error("--decode_max_frames must be at least --max_frames")
    if not 0 <= args.default_score <= 1:
        parser.error("--default_score must be between 0 and 1")
    if args.likelihood_temperature <= 0:
        parser.error("--likelihood_temperature must be positive")
    if args.threshold_batch_size <= 0:
        parser.error("--threshold_batch_size must be positive")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint_interval must be positive")
    if args.yes_candidate == args.no_candidate:
        parser.error("--yes_candidate and --no_candidate must differ")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    names = split_for_job(
        load_video_names(args.video_dir, args.index_file, args.video_list),
        args.num_jobs,
        args.job_index,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(
        args.model_path,
        args.device,
        args.attn_implementation,
    )
    LOGGER.info(
        "Experiment=%s global=%d center=%d center_seconds=%.3f "
        "sample_fps=%.3f max_frames=%d optimized=%s",
        EXPERIMENT,
        args.global_frames,
        args.center_frames,
        args.center_seconds,
        args.sample_fps,
        args.max_frames,
        args.optimized,
    )
    for name in names:
        try:
            process_video(args, model, processor, name)
        except Exception:
            LOGGER.exception("Failed to initialize/process video %s; continuing", name)


if __name__ == "__main__":
    main()
