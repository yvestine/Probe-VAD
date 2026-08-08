"""E0 temporal-order ablation on raw video clips.

The scoring model, cumulative thresholds, PAVA projection, and aggregation are
kept identical to ``video_cumulative_score.py``.  Only the already-decoded
frame sequence is changed before the shared-vision likelihood call.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import random
from pathlib import Path

from tqdm import tqdm

from src.video_cumulative_score import CUMULATIVE_PROMPT
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
    preload_video_conversation,
    resolve_video_path,
    split_for_job,
)

LOGGER = logging.getLogger("video_temporal_order_ablation")


def _stable_seed(global_seed: int, name: str, center_frame: int) -> int:
    payload = f"{global_seed}:{name}:{center_frame}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def _transform_loaded_video(
    conversation,
    mode: str,
    seed: int,
    video_name: str,
    center_frame: int,
) -> None:
    video_item = conversation[1]["content"][0]
    frames = list(video_item["video"])
    timestamps = list(video_item.get("timestamps", range(len(frames))))
    if not frames:
        raise RuntimeError("decoded clip contains no frames")
    if len(timestamps) != len(frames):
        timestamps = list(range(len(frames)))

    if mode == "shuffle":
        order = list(range(len(frames)))
        random.Random(_stable_seed(seed, video_name, center_frame)).shuffle(order)
        frames = [frames[index] for index in order]
        timestamps = [timestamps[index] for index in order]
    elif mode == "single":
        index = len(frames) // 2
        frames = [frames[index]]
        timestamps = [timestamps[index]]
    elif mode != "order":
        raise ValueError(f"unknown temporal mode: {mode}")

    video_item["video"] = frames
    video_item["timestamps"] = timestamps
    video_item["num_frames"] = len(frames)


def process_video(args, model, processor, name: str) -> None:
    video_path = resolve_video_path(args.video_dir, name)
    stem = output_stem(name)
    output_path = args.output_dir / f"{stem}.json"
    details_path = args.output_dir / "_threshold_details" / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    info = get_video_info(video_path)
    windows = list(iter_video_windows(info, args.frame_interval, args.window_seconds))
    expected = [str(window.center_frame) for window in windows]
    scores = load_json_dict(output_path) if args.resume else {}
    details = load_json_dict(details_path) if args.resume else {}
    errors = load_json_dict(error_path) if args.resume else {}
    if (
        args.resume
        and is_complete(scores, expected)
        and (not args.save_threshold_details or set(details) == set(expected))
        and not errors
    ):
        return

    LOGGER.info("%s: %d windows, mode=%s", stem, len(windows), args.temporal_mode)
    dirty = 0

    def checkpoint() -> None:
        nonlocal dirty
        if dirty == 0:
            return
        atomic_write_json(output_path, scores)
        if args.save_threshold_details:
            atomic_write_json(details_path, details)
        if errors:
            atomic_write_json(error_path, errors)
        elif error_path.exists():
            error_path.unlink()
        dirty = 0

    try:
        for window in tqdm(windows, desc=stem, unit="window"):
            key = str(window.center_frame)
            if (
                args.resume
                and key in scores
                and key not in errors
                and (not args.save_threshold_details or key in details)
            ):
                continue
            try:
                # Decode and preprocess the clip once, then alter only its frame
                # order/count before all ten threshold questions share vision.
                base = build_threshold_conversation(
                    video_path,
                    window,
                    args.sample_fps,
                    args.max_frames,
                    args.prompt,
                    float(CUMULATIVE_THRESHOLDS[0]),
                    precise_time=args.precise_time,
                )
                loaded = preload_video_conversation(
                    processor,
                    base,
                    video_path,
                    window,
                    args.sample_fps,
                    args.max_frames,
                    args.precise_time,
                )
                _transform_loaded_video(
                    loaded,
                    args.temporal_mode,
                    args.shuffle_seed,
                    stem,
                    window.center_frame,
                )
                conversations = []
                for threshold in CUMULATIVE_THRESHOLDS:
                    template = build_threshold_conversation(
                        video_path,
                        window,
                        args.sample_fps,
                        args.max_frames,
                        args.prompt,
                        float(threshold),
                        precise_time=args.precise_time,
                    )
                    current = copy.copy(loaded)
                    current[1] = copy.copy(loaded[1])
                    current[1]["content"] = list(loaded[1]["content"])
                    current[1]["content"][1] = dict(template[1]["content"][1])
                    conversations.append(current)

                probabilities, threshold_details = (
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
                score, adjusted, violations = cumulative_score_from_tail_probabilities(
                    probabilities,
                    monotonic_projection=args.monotonic_projection,
                )
                scores[key] = score
                if args.save_threshold_details:
                    details[key] = {
                        "thresholds": [
                            {"threshold": threshold, **detail}
                            for threshold, detail in zip(
                                CUMULATIVE_THRESHOLDS, threshold_details
                            )
                        ],
                        "monotonic_tail_probabilities": adjusted,
                        "monotonic_violations": violations,
                        "temporal_mode": args.temporal_mode,
                        "score": score,
                    }
                errors.pop(key, None)
            except Exception as exc:
                scores[key] = args.default_score
                errors[key] = {"raw_output": "", "error": repr(exc)}
                LOGGER.exception("%s frame=%s failed", stem, key)
            dirty += 1
            if dirty >= args.checkpoint_interval:
                checkpoint()
    finally:
        checkpoint()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_dir", type=Path, required=True)
    p.add_argument("--index_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--temporal_mode", choices=["order", "shuffle", "single"], required=True)
    p.add_argument("--shuffle_seed", type=int, default=17)
    p.add_argument("--frame_interval", type=int, default=16)
    p.add_argument("--window_seconds", type=float, default=10.0)
    p.add_argument("--sample_fps", type=float, default=2.0)
    p.add_argument("--max_frames", type=int, default=10)
    p.add_argument("--precise_time", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--yes_candidate", default="YES")
    p.add_argument("--no_candidate", default="NO")
    p.add_argument("--likelihood_temperature", type=float, default=1.0)
    p.add_argument("--threshold_batch_size", type=int, default=10)
    p.add_argument("--prefix_cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--monotonic_projection", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--default_score", type=float, default=0.5)
    p.add_argument("--checkpoint_interval", type=int, default=20)
    p.add_argument("--save_threshold_details", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--prompt", default=CUMULATIVE_PROMPT)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_jobs", type=int, default=1)
    p.add_argument("--job_index", type=int, default=0)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    names = split_for_job(
        load_video_names(args.video_dir, args.index_file),
        args.num_jobs,
        args.job_index,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.model_path, None, args.attn_implementation)
    for name in names:
        try:
            process_video(args, model, processor, name)
        except Exception:
            LOGGER.exception("Failed to initialize/process video %s", name)


if __name__ == "__main__":
    main()
