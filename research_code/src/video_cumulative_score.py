"""Direct video scoring with cumulative Yes/No threshold likelihoods.

This entry point observes raw video clips only. It does not generate captions,
select suspicious intervals, generate anomaly tags, or perform refinement.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from src.video_score_utils import (
    CUMULATIVE_THRESHOLDS,
    DEFAULT_MODEL_PATH,
    atomic_write_json,
    build_threshold_conversation,
    cumulative_threshold_likelihood_optimized,
    cumulative_score_from_tail_probabilities,
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

LOGGER = logging.getLogger("video_cumulative_score")
CUMULATIVE_PROMPT = (
    "You are a careful video anomaly detector for public surveillance scenes. "
    "Judge only visible evidence in the current video segment. Use an ordered "
    "anomaly-severity scale from 0 to 1: 0 means ordinary safe activity; 0.5 "
    "means clearly concerning or plausibly anomalous activity; and 1 means "
    "unmistakably severe, dangerous, violent, criminal, or emergency activity. "
    "Intermediate thresholds preserve this order. Do not infer events that are "
    "not visually supported."
)


def process_video(args, model, processor, name: str) -> None:
    video_path = resolve_video_path(args.video_dir, name)
    stem = output_stem(name)
    output_path = args.output_dir / f"{stem}.json"
    detail_path = args.output_dir / "_threshold_details" / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    info = get_video_info(video_path)
    windows = list(iter_video_windows(info, args.frame_interval, args.window_seconds))
    expected = [str(window.center_frame) for window in windows]
    scores = load_json_dict(output_path) if args.resume else {}
    details = (
        load_json_dict(detail_path)
        if args.resume and args.save_threshold_details
        else {}
    )
    errors = load_json_dict(error_path) if args.resume else {}
    if (
        args.resume
        and is_complete(scores, expected)
        and (
            not args.save_threshold_details
            or set(details) == set(expected)
        )
        and not errors
    ):
        LOGGER.info("Skipping complete video %s", stem)
        return

    LOGGER.info(
        "Video %s: %d frames, %.3f FPS, %d windows",
        stem, info.frame_count, info.fps, len(windows),
    )
    dirty_windows = 0

    def checkpoint() -> None:
        nonlocal dirty_windows
        if dirty_windows == 0:
            return
        atomic_write_json(output_path, scores)
        if args.save_threshold_details:
            atomic_write_json(detail_path, details)
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
                    or key in details
                )
                and key not in errors
            ):
                continue
            probabilities = []
            threshold_details = []
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
                        threshold_details = [
                            {
                                "threshold": threshold,
                                **detail,
                            }
                            for threshold, detail in zip(
                                CUMULATIVE_THRESHOLDS, optimized_details
                            )
                        ]
                else:
                    for threshold, conversation in zip(
                        CUMULATIVE_THRESHOLDS, conversations
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
                            threshold_details.append(
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
                    details[key] = {
                        "thresholds": threshold_details,
                        "monotonic_tail_probabilities": adjusted,
                        "monotonic_violations": violations,
                        "score": score,
                    }
                errors.pop(key, None)
                LOGGER.info(
                    "%s frame=%s time=%.3f-%.3f score=%.6f "
                    "tail_raw=%s tail_monotonic=%s violations=%d",
                    stem,
                    key,
                    window.start_time,
                    window.end_time,
                    score,
                    [round(value, 6) for value in probabilities],
                    [round(value, 6) for value in adjusted],
                    violations,
                )
            except Exception as exc:
                scores[key] = args.default_score
                errors[key] = {"raw_output": "", "error": repr(exc)}
                LOGGER.exception(
                    "%s frame=%s time=%.3f-%.3f failed; using default %.3f",
                    stem,
                    key,
                    window.start_time,
                    window.end_time,
                    args.default_score,
                )
            dirty_windows += 1
            if dirty_windows >= args.checkpoint_interval:
                checkpoint()
    finally:
        # SIGINT/KeyboardInterrupt loses at most the currently executing model
        # call; all completed windows in the in-memory batch are checkpointed.
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
    parser.add_argument(
        "--precise_time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use slower output-side FFmpeg trimming. Enable for videos with "
            "non-zero stream start timestamps, such as MSAD."
        ),
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
        help="Length-normalize complete YES/NO candidate sequence likelihoods.",
    )
    parser.add_argument("--likelihood_temperature", type=float, default=1.0)
    parser.add_argument(
        "--optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Share video decoding/vision encoding and batch threshold logits. "
            "Disable only for numerical reference validation."
        ),
    )
    parser.add_argument(
        "--threshold_batch_size",
        type=int,
        default=10,
        help="Number of threshold suffixes evaluated together (default: 10).",
    )
    parser.add_argument(
        "--prefix_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the common multimodal/text prefix through the LM KV cache.",
    )
    parser.add_argument(
        "--monotonic_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project the ten tail probabilities onto a non-increasing sequence.",
    )
    parser.add_argument("--default_score", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Atomically save after this many newly processed windows (default: 20).",
    )
    parser.add_argument(
        "--save_threshold_details",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist per-threshold diagnostics; disabled by default to reduce I/O.",
    )
    parser.add_argument("--prompt", default=CUMULATIVE_PROMPT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.default_score <= 1:
        parser.error("--default_score must be between 0 and 1")
    if args.likelihood_temperature <= 0:
        parser.error("--likelihood_temperature must be positive")
    if args.threshold_batch_size <= 0:
        parser.error("--threshold_batch_size must be positive")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint_interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    names = split_for_job(
        load_video_names(args.video_dir, args.index_file, args.video_list),
        args.num_jobs, args.job_index,
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
