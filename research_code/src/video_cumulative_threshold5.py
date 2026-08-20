"""Actual five-threshold cumulative likelihood scoring.

This is an independent ablation of E0.  It keeps the raw-video input,
VideoLLaMA3 checkpoint, windowing, sampling, YES/NO likelihood and PAVA, but
asks only the five thresholds 0.2, 0.4, 0.6, 0.8 and 1.0.  Existing E0 output
directories are never modified.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from src.video_score_utils import (
    DEFAULT_MODEL_PATH,
    atomic_write_json,
    clear_cuda_cache,
    build_threshold_conversation,
    cumulative_threshold_likelihood_optimized,
    decreasing_isotonic_projection,
    get_video_info,
    is_complete,
    load_json_dict,
    load_model,
    load_video_names,
    output_stem,
    resolve_video_path,
    split_for_job,
    threshold_yes_no_likelihood,
    iter_video_windows,
)

LOGGER = logging.getLogger("video_cumulative_threshold5")
DEFAULT_THRESHOLDS = (0.2, 0.4, 0.6, 0.8, 1.0)
CUMULATIVE_PROMPT = (
    "You are a careful video anomaly detector for public surveillance scenes. "
    "Judge only visible evidence in the current video segment. Use an ordered "
    "anomaly-severity scale from 0 to 1: 0 means ordinary safe activity; 0.5 "
    "means clearly concerning or plausibly anomalous activity; and 1 means "
    "unmistakably severe, dangerous, violent, criminal, or emergency activity. "
    "Intermediate thresholds preserve this order. Do not infer events that are "
    "not visually supported."
)


def five_threshold_score(probabilities: list[float], monotonic: bool):
    if len(probabilities) != 5:
        raise ValueError(f"expected five tail probabilities, got {len(probabilities)}")
    raw = [min(1.0, max(0.0, float(value))) for value in probabilities]
    violations = sum(raw[i] < raw[i + 1] for i in range(len(raw) - 1))
    adjusted = decreasing_isotonic_projection(raw) if monotonic else raw
    return float(sum(adjusted) / len(adjusted)), adjusted, violations


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
    details = load_json_dict(detail_path) if args.resume else {}
    errors = load_json_dict(error_path) if args.resume else {}
    if args.resume and is_complete(scores, expected) and (
        not args.save_threshold_details or set(details) == set(expected)
    ) and not errors:
        return

    dirty = 0

    def checkpoint() -> None:
        nonlocal dirty
        if dirty == 0:
            return
        atomic_write_json(output_path, scores)
        if args.save_threshold_details:
            atomic_write_json(detail_path, details)
        if errors:
            atomic_write_json(error_path, errors)
        elif error_path.exists():
            error_path.unlink()
        dirty = 0
        clear_cuda_cache()

    try:
        for window in tqdm(windows, desc=stem, unit="window", leave=False):
            key = str(window.center_frame)
            if (
                args.resume
                and key in scores
                and (not args.save_threshold_details or key in details)
                and key not in errors
            ):
                continue
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
                    for threshold in args.thresholds
                ]
                if args.optimized:
                    probabilities, raw_details = cumulative_threshold_likelihood_optimized(
                        model,
                        processor,
                        conversations,
                        yes_candidate=args.yes_candidate,
                        no_candidate=args.no_candidate,
                        temperature=args.likelihood_temperature,
                        threshold_batch_size=args.threshold_batch_size,
                        prefix_cache=args.prefix_cache,
                    )
                else:
                    probabilities = []
                    raw_details = []
                    for threshold, conversation in zip(args.thresholds, conversations):
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
                        raw_details.append(
                            {
                                "threshold": threshold,
                                "yes_loglik": yes_ll,
                                "no_loglik": no_ll,
                                "tail_probability": probability,
                            }
                        )
                score, adjusted, violations = five_threshold_score(
                    probabilities, args.monotonic_projection
                )
                scores[key] = score
                if args.save_threshold_details:
                    details[key] = {
                        "thresholds": [
                            {"threshold": float(t), **detail}
                            for t, detail in zip(args.thresholds, raw_details)
                        ],
                        "monotonic_tail_probabilities": adjusted,
                        "monotonic_violations": int(violations),
                        "score": score,
                    }
                errors.pop(key, None)
            except Exception as exc:
                scores[key] = args.default_score
                errors[key] = {"raw_output": "", "error": repr(exc)}
                LOGGER.exception("%s center=%s failed", stem, key)
            dirty += 1
            if dirty >= args.checkpoint_interval:
                checkpoint()
    finally:
        checkpoint()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--index_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--precise_time", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--yes_candidate", default="YES")
    parser.add_argument("--no_candidate", default="NO")
    parser.add_argument("--length_normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--likelihood_temperature", type=float, default=1.0)
    parser.add_argument("--optimized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold_batch_size", type=int, default=5)
    parser.add_argument("--prefix_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monotonic_projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_threshold_details", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--default_score", type=float, default=0.5)
    parser.add_argument("--prompt", default=CUMULATIVE_PROMPT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--log_level", default="WARNING")
    args = parser.parse_args()
    args.thresholds = tuple(float(value) for value in args.thresholds.split(",") if value.strip())
    if len(args.thresholds) != 5:
        parser.error("--thresholds must contain exactly five values")
    if args.likelihood_temperature <= 0 or args.threshold_batch_size <= 0:
        parser.error("temperature and threshold_batch_size must be positive")
    if args.num_jobs <= 0 or not 0 <= args.job_index < args.num_jobs:
        parser.error("invalid job_index/num_jobs")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    names = split_for_job(
        load_video_names(args.video_dir, args.index_file), args.num_jobs, args.job_index
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.model_path, args.device, args.attn_implementation)
    for name in names:
        process_video(args, model, processor, name)


if __name__ == "__main__":
    main()
