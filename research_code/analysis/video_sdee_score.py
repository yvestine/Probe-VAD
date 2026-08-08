"""E5-SDEE direct scoring with five RGB states and five D1 motion maps.

Only the ten visual inputs differ from E0.  VideoLLaMA3, the ten cumulative
YES/NO thresholds, likelihood computation, PAVA, score integration, sliding
windows, output JSON, and evaluation interface remain unchanged.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from src.sdee_motion import construct_sdee_sequence
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
)

LOGGER = logging.getLogger("video_sdee_score")

SDEE_INPUT_INSTRUCTION = (
    " The ten visual inputs are ordered as S1, M1, S2, M2, S3, M3, S4, M4, "
    "S5, M5 across five consecutive temporal segments. Each S image is an RGB "
    "state frame near the segment center. Each M image is a first-order motion "
    "evidence map for the same segment: red indicates compensated appearance "
    "change, green indicates residual displacement magnitude, and blue "
    "indicates their spatial overlap. Strong motion alone does not imply an "
    "anomaly. Interpret each motion map together with its paired state frame "
    "and judge only visually supported behavior."
)

SDEE_NOALIGN_INPUT_INSTRUCTION = (
    " The ten visual inputs are ordered as S1, M1, S2, M2, S3, M3, S4, M4, "
    "S5, M5 across five consecutive temporal segments. Each S image is an RGB "
    "state frame near the segment center. Each M image is a first-order motion "
    "evidence map for the same segment without global camera-motion "
    "compensation: red indicates appearance change, green indicates "
    "displacement magnitude, and blue indicates their spatial overlap. Strong "
    "motion alone does not imply an anomaly. Interpret each motion map together "
    "with its paired state frame and judge only visually supported behavior."
)


def experiment_name(motion_mode: str, shuffle_motion: bool) -> str:
    if shuffle_motion:
        return "e5_sdee_d1_shuffle"
    if motion_mode == "noalign":
        return "e5_sdee_d1_noalign"
    return "e5_sdee_d1"


def e5_prompt(motion_mode: str) -> str:
    instruction = (
        SDEE_NOALIGN_INPUT_INSTRUCTION
        if motion_mode == "noalign"
        else SDEE_INPUT_INSTRUCTION
    )
    return CUMULATIVE_PROMPT + instruction


def load_sdee_window(
    processor,
    video_path: Path,
    window: VideoWindow,
    motion_fps: float,
    decode_max_frames: int,
    temporal_bins: int,
    motion_mode: str,
    shuffle_motion: bool,
    precise_time: bool,
) -> tuple[list[Any], list[float], dict[str, Any]]:
    duration = max(0.0, window.end_time - window.start_time)
    expected = int(math.ceil(duration * motion_fps)) + 2
    candidate_limit = max(decode_max_frames, expected, temporal_bins * 2)
    frames, timestamps = processor.load_video(
        video_path=str(video_path),
        start_time=window.start_time,
        end_time=window.end_time,
        fps=motion_fps,
        max_frames=candidate_limit,
        precise_time=precise_time,
    )
    if len(frames) != len(timestamps):
        raise RuntimeError(
            "decoder returned mismatched frames/timestamps: "
            f"{len(frames)} != {len(timestamps)}"
        )
    sequence, sequence_times, details = construct_sdee_sequence(
        frames,
        timestamps,
        window.start_time,
        window.end_time,
        temporal_bins=temporal_bins,
        align_global_motion=motion_mode == "aligned",
        shuffle_motion=shuffle_motion,
    )
    details.update(
        {
            "candidate_count": len(frames),
            "candidate_timestamps": [float(value) for value in timestamps],
            "motion_fps": motion_fps,
            "decode_max_frames": candidate_limit,
        }
    )
    return sequence, sequence_times, details


def build_preloaded_conversation(
    video_path: Path,
    window: VideoWindow,
    prompt: str,
    threshold: float,
    sequence: Sequence[Any],
    sequence_times: Sequence[float],
    precise_time: bool,
) -> list[dict[str, Any]]:
    conversation = build_threshold_conversation(
        video_path,
        window,
        sample_fps=1.0,
        max_frames=len(sequence),
        prompt=prompt,
        threshold=threshold,
        precise_time=precise_time,
    )
    video_item = conversation[1]["content"][0]
    video_item["video"] = list(sequence)
    video_item["num_frames"] = len(sequence)
    video_item["timestamps"] = [float(value) for value in sequence_times]
    return conversation


def process_video(args, model, processor, name: str) -> None:
    video_path = resolve_video_path(args.video_dir, name)
    stem = output_stem(name)
    info = get_video_info(video_path)
    windows = list(
        iter_video_windows(
            info,
            frame_interval=args.frame_interval,
            window_seconds=args.window_seconds,
        )
    )
    expected = [str(window.center_frame) for window in windows]
    output_path = args.output_dir / f"{stem}.json"
    threshold_path = args.output_dir / "_threshold_details" / f"{stem}.json"
    motion_path = args.output_dir / "_motion_details" / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    scores = load_json_dict(output_path) if args.resume else {}
    threshold_details = (
        load_json_dict(threshold_path)
        if args.resume and args.save_threshold_details
        else {}
    )
    motion_details = (
        load_json_dict(motion_path)
        if args.resume and args.save_motion_details
        else {}
    )
    errors = load_json_dict(error_path) if args.resume else {}
    if (
        args.resume
        and is_complete(scores, expected)
        and (
            not args.save_threshold_details
            or is_complete(threshold_details, expected)
        )
        and (
            not args.save_motion_details
            or is_complete(motion_details, expected)
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
        if args.save_threshold_details:
            atomic_write_json(threshold_path, threshold_details)
        if args.save_motion_details:
            atomic_write_json(motion_path, motion_details)
        if errors:
            atomic_write_json(error_path, errors)
        elif error_path.exists():
            error_path.unlink()
        dirty = 0

    experiment = experiment_name(args.motion_mode, args.shuffle_motion)
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
                    not args.save_motion_details
                    or key in motion_details
                )
                and key not in errors
            ):
                continue
            try:
                sequence, sequence_times, evidence_details = load_sdee_window(
                    processor,
                    video_path,
                    window,
                    motion_fps=args.motion_fps,
                    decode_max_frames=args.decode_max_frames,
                    temporal_bins=args.temporal_bins,
                    motion_mode=args.motion_mode,
                    shuffle_motion=args.shuffle_motion,
                    precise_time=args.precise_time,
                )
                conversations = [
                    build_preloaded_conversation(
                        video_path,
                        window,
                        args.prompt,
                        threshold,
                        sequence,
                        sequence_times,
                        args.precise_time,
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
                score, adjusted, violations = (
                    cumulative_score_from_tail_probabilities(
                        probabilities,
                        monotonic_projection=True,
                    )
                )
                scores[key] = score
                if args.save_threshold_details:
                    threshold_details[key] = {
                        "experiment": experiment,
                        "thresholds": [
                            {"threshold": threshold, **detail}
                            for threshold, detail in zip(
                                CUMULATIVE_THRESHOLDS,
                                raw_details,
                            )
                        ],
                        "monotonic_tail_probabilities": adjusted,
                        "monotonic_violations": violations,
                        "score": score,
                    }
                if args.save_motion_details:
                    motion_details[key] = {
                        "experiment": experiment,
                        "center_frame": window.center_frame,
                        "score": score,
                        **evidence_details,
                    }
                errors.pop(key, None)
                LOGGER.info(
                    "%s experiment=%s frame=%s time=%.3f-%.3f "
                    "score=%.6f violations=%d alignment=%s shuffle=%s",
                    stem,
                    experiment,
                    key,
                    window.start_time,
                    window.end_time,
                    score,
                    violations,
                    args.motion_mode,
                    args.shuffle_motion,
                )
            except Exception as exc:
                scores[key] = args.default_score
                errors[key] = {
                    "error": repr(exc),
                    "center_frame": window.center_frame,
                    "window": [window.start_time, window.end_time],
                }
                LOGGER.exception(
                    "%s frame=%s E5-SDEE failed; using default %.3f",
                    stem,
                    key,
                    args.default_score,
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
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument("--temporal_bins", type=int, default=5)
    parser.add_argument("--motion_fps", type=float, default=8.0)
    parser.add_argument("--decode_max_frames", type=int, default=96)
    parser.add_argument(
        "--motion_mode",
        choices=("aligned", "noalign"),
        default="aligned",
    )
    parser.add_argument(
        "--shuffle_motion",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    parser.add_argument("--likelihood_temperature", type=float, default=1.0)
    parser.add_argument("--threshold_batch_size", type=int, default=10)
    parser.add_argument(
        "--prefix_cache",
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
        "--save_motion_details",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if args.frame_interval <= 0 or args.window_seconds <= 0:
        parser.error("frame interval and window seconds must be positive")
    if args.temporal_bins != 5:
        parser.error("E5-SDEE requires exactly five temporal bins")
    if args.motion_fps <= 0 or args.decode_max_frames < 10:
        parser.error("invalid motion decoding parameters")
    if args.threshold_batch_size <= 0 or args.checkpoint_interval <= 0:
        parser.error("batch and checkpoint intervals must be positive")
    if not 0 <= args.default_score <= 1:
        parser.error("default score must lie in [0, 1]")
    if args.prompt is None:
        args.prompt = e5_prompt(args.motion_mode)
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
