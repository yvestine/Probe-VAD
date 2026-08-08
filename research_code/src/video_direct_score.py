"""Directly score raw video windows with VideoLLaMA3 (no captions)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from src.video_score_utils import (
    DEFAULT_MODEL_PATH,
    atomic_write_json,
    build_video_conversation,
    get_video_info,
    is_complete,
    iter_video_windows,
    load_json_dict,
    load_model,
    load_video_names,
    output_stem,
    preload_video_conversation,
    resolve_video_path,
    score_conversation,
    split_for_job,
)

LOGGER = logging.getLogger("video_direct_score")
DIRECT_PROMPT = (
    "You are a careful video anomaly detector for public surveillance scenes. "
    "Judge only the visible behavior in the current video segment. A score of 0 "
    "means ordinary, safe activity and 1 means clearly abnormal, dangerous, "
    "violent, criminal, or emergency activity. Intermediate values represent "
    "uncertainty or severity. Do not infer events that are not visually supported."
)


def process_video(args, model, processor, name: str) -> None:
    video_path = resolve_video_path(args.video_dir, name)
    stem = output_stem(name)
    output_path = args.output_dir / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    info = get_video_info(video_path)
    windows = list(iter_video_windows(info, args.frame_interval, args.window_seconds))
    expected = [str(window.center_frame) for window in windows]
    scores = load_json_dict(output_path) if args.resume else {}
    errors = load_json_dict(error_path) if args.resume else {}
    if args.resume and is_complete(scores, expected) and not errors:
        LOGGER.info("Skipping complete video %s", stem)
        return

    LOGGER.info(
        "Video %s: %d frames, %.3f FPS, %d windows",
        stem,
        info.frame_count,
        info.fps,
        len(windows),
    )
    for window in tqdm(windows, desc=stem, unit="window"):
        key = str(window.center_frame)
        if args.resume and key in scores and key not in errors:
            continue
        conversation = build_video_conversation(
            video_path,
            window,
            args.sample_fps,
            args.max_frames,
            args.prompt,
            args.score_mode,
            precise_time=args.precise_time,
        )
        try:
            conversation = preload_video_conversation(
                processor,
                conversation,
                video_path,
                window,
                args.sample_fps,
                args.max_frames,
                args.precise_time,
            )
            score, raw, parse_error = score_conversation(
                model,
                processor,
                conversation,
                args.score_mode,
                args.default_score,
                args.max_new_tokens,
                args.normal_candidate,
                args.anomalous_candidate,
                args.length_normalize,
                args.likelihood_temperature,
            )
            scores[key] = score
            if parse_error:
                errors[key] = {"raw_output": raw, "error": parse_error}
                LOGGER.warning(
                    "%s frame=%s time=%.3f-%.3f raw=%r score=%.4f error=%s",
                    stem, key, window.start_time, window.end_time, raw, score, parse_error,
                )
            else:
                errors.pop(key, None)
                LOGGER.info(
                    "%s frame=%s time=%.3f-%.3f raw=%r score=%.4f",
                    stem, key, window.start_time, window.end_time, raw, score,
                )
        except Exception as exc:  # preserve progress for isolated decoding/model failures
            scores[key] = args.default_score
            errors[key] = {"raw_output": "", "error": repr(exc)}
            LOGGER.exception(
                "%s frame=%s time=%.3f-%.3f failed; using default %.3f",
                stem, key, window.start_time, window.end_time, args.default_score,
            )
        atomic_write_json(output_path, scores)
        if errors:
            atomic_write_json(error_path, errors)
        elif error_path.exists():
            error_path.unlink()


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
    parser.add_argument(
        "--precise_time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use exact video timestamps when decoding each window. This is "
            "recommended for MSAD videos with non-zero or irregular stream "
            "timestamps."
        ),
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument(
        "--score_mode",
        choices=[
            "generated",
            "likelihood",
            "likelihood_optimized",
            "binary_likelihood",
        ],
        default="generated",
        help=(
            "generated: discrete generation; likelihood: expected score over "
            "11 bracketed candidates; likelihood_optimized: shared-vision "
            "batched 11-candidate likelihood; binary_likelihood: "
            "NORMAL/ANOMALOUS probability"
        ),
    )
    parser.add_argument("--default_score", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--normal_candidate", default="NORMAL")
    parser.add_argument("--anomalous_candidate", default="ANOMALOUS")
    parser.add_argument("--length_normalize", action="store_true")
    parser.add_argument(
        "--likelihood_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature over the 11 ordinal candidate log-likelihoods.",
    )
    parser.add_argument("--prompt", default=DIRECT_PROMPT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.default_score <= 1:
        parser.error("--default_score must be between 0 and 1")
    if args.likelihood_temperature <= 0:
        parser.error("--likelihood_temperature must be positive")
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
        args.model_path, args.device, args.attn_implementation
    )
    for name in names:
        try:
            process_video(args, model, processor, name)
        except Exception:
            LOGGER.exception("Failed to initialize/process video %s; continuing", name)


if __name__ == "__main__":
    main()
