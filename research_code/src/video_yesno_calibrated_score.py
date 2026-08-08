"""Direct video scoring with polarity-calibrated YES/NO likelihoods.

This is an isolated E3 ablation entry point.  For every anomaly threshold it
asks both a proposition and its strict complement:

* forward: is anomaly severity at least the threshold?
* reverse: is anomaly severity strictly below the threshold?

The anomaly-oriented margins are averaged before sigmoid:

    m1 = logit(YES_forward) - logit(NO_forward)
    m2 = logit(NO_reverse) - logit(YES_reverse)
    p  = sigmoid((m1 + m2) / (2 * temperature))

It observes raw video clips only and does not generate captions, anomaly tags,
refinements, or fused scores.  Existing E0/E1/E2 implementations are not
modified or used as writable outputs.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

LOGGER = logging.getLogger("video_yesno_calibrated_score")
EXPERIMENT = "e3"


def stable_sigmoid(value: float) -> float:
    """Return a numerically stable scalar sigmoid."""
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def polarity_calibrated_probability(
    forward_yes_logit: float,
    forward_no_logit: float,
    reverse_yes_logit: float,
    reverse_no_logit: float,
    *,
    temperature: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Return E3 probability and its anomaly-oriented margins.

    The reverse prompt asks whether severity is strictly below the threshold,
    so NO is the anomaly-oriented answer in that prompt.  Margin averaging is
    intentionally performed before sigmoid.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    margin_1 = float(forward_yes_logit) - float(forward_no_logit)
    margin_2 = float(reverse_no_logit) - float(reverse_yes_logit)
    calibrated_margin = (margin_1 + margin_2) / 2.0
    probability = stable_sigmoid(calibrated_margin / temperature)
    return probability, calibrated_margin, margin_1, margin_2


def build_yesno_polarity_conversation(
    video_path: Path,
    window: VideoWindow,
    sample_fps: float,
    max_frames: int,
    prompt: str,
    threshold: float,
    *,
    reverse: bool,
    precise_time: bool = False,
) -> List[Dict[str, Any]]:
    """Build an E0-identical forward prompt or its strict complement."""
    if not reverse:
        return build_threshold_conversation(
            video_path,
            window,
            sample_fps,
            max_frames,
            prompt,
            threshold,
            precise_time=precise_time,
        )

    question = (
        f"On the severity scale defined by the system, is the anomaly severity "
        f"visible in this video segment strictly below {threshold:.1f}? "
        "Judge only the current video's visual evidence. Answer exactly YES or NO."
    )
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": str(video_path),
                        "fps": sample_fps,
                        "start_time": window.start_time,
                        "end_time": window.end_time,
                        "max_frames": max_frames,
                        "precise_time": precise_time,
                    },
                },
                {"type": "text", "text": question},
            ],
        },
    ]


def validate_single_token_candidates(
    processor,
    yes_candidate: str = "YES",
    no_candidate: str = "NO",
) -> Tuple[int, int]:
    """Validate the exact continuation tokens used by optimized E3 scoring."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor does not expose a tokenizer")
    yes_ids = tokenizer(yes_candidate, add_special_tokens=False).input_ids
    no_ids = tokenizer(no_candidate, add_special_tokens=False).input_ids
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(
            "E3 optimized scoring requires one-token candidates; "
            f"{yes_candidate!r}={yes_ids}, {no_candidate!r}={no_ids}"
        )
    if yes_ids[0] == no_ids[0]:
        raise ValueError("YES and NO candidates must map to different token IDs")
    return int(yes_ids[0]), int(no_ids[0])


def _candidate_logits_reference(
    model,
    processor,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    yes_candidate: str,
    no_candidate: str,
) -> List[Dict[str, float]]:
    """Slow per-prompt reference implementation for numerical validation."""
    diagnostics: List[Dict[str, float]] = []
    for conversation in conversations:
        _, yes_logit, no_logit = threshold_yes_no_likelihood(
            model,
            processor,
            conversation,
            yes_candidate=yes_candidate,
            no_candidate=no_candidate,
            length_normalize=False,
            temperature=1.0,
        )
        diagnostics.append(
            {
                "yes_logit": float(yes_logit),
                "no_logit": float(no_logit),
            }
        )
    return diagnostics


def _candidate_logits_optimized(
    model,
    processor,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    yes_candidate: str,
    no_candidate: str,
    threshold_batch_size: int,
    prefix_cache: bool,
) -> List[Dict[str, float]]:
    """Evaluate all twenty E3 prompts with one shared visual encoding."""
    _, raw_details = cumulative_threshold_likelihood_optimized(
        model,
        processor,
        conversations,
        yes_candidate=yes_candidate,
        no_candidate=no_candidate,
        temperature=1.0,
        threshold_batch_size=threshold_batch_size,
        prefix_cache=prefix_cache,
    )
    return [
        {
            # The common utility reports log-softmax values.  Their difference
            # is exactly the corresponding raw-logit difference.
            "yes_logit": float(detail["yes_loglik"]),
            "no_logit": float(detail["no_loglik"]),
        }
        for detail in raw_details
    ]


def polarity_calibrated_tail_probabilities(
    model,
    processor,
    forward_conversations: Sequence[Sequence[Mapping[str, Any]]],
    reverse_conversations: Sequence[Sequence[Mapping[str, Any]]],
    *,
    yes_candidate: str,
    no_candidate: str,
    temperature: float,
    optimized: bool,
    threshold_batch_size: int,
    prefix_cache: bool,
) -> Tuple[List[float], List[Dict[str, float]]]:
    """Compute ten E3 tail probabilities and per-threshold diagnostics."""
    expected_count = len(CUMULATIVE_THRESHOLDS)
    if len(forward_conversations) != expected_count:
        raise ValueError(
            f"expected {expected_count} forward prompts, "
            f"got {len(forward_conversations)}"
        )
    if len(reverse_conversations) != expected_count:
        raise ValueError(
            f"expected {expected_count} reverse prompts, "
            f"got {len(reverse_conversations)}"
        )

    all_conversations = list(forward_conversations) + list(reverse_conversations)
    if optimized:
        raw_logits = _candidate_logits_optimized(
            model,
            processor,
            all_conversations,
            yes_candidate,
            no_candidate,
            threshold_batch_size,
            prefix_cache,
        )
    else:
        raw_logits = _candidate_logits_reference(
            model,
            processor,
            all_conversations,
            yes_candidate,
            no_candidate,
        )

    forward_logits = raw_logits[:expected_count]
    reverse_logits = raw_logits[expected_count:]
    probabilities: List[float] = []
    diagnostics: List[Dict[str, float]] = []
    for forward, reverse in zip(forward_logits, reverse_logits):
        probability, margin, margin_1, margin_2 = (
            polarity_calibrated_probability(
                forward["yes_logit"],
                forward["no_logit"],
                reverse["yes_logit"],
                reverse["no_logit"],
                temperature=temperature,
            )
        )
        probabilities.append(probability)
        diagnostics.append(
            {
                "forward_yes_logit": forward["yes_logit"],
                "forward_no_logit": forward["no_logit"],
                "reverse_yes_logit": reverse["yes_logit"],
                "reverse_no_logit": reverse["no_logit"],
                "margin_1": margin_1,
                "margin_2": margin_2,
                "calibrated_margin": margin,
                "tail_probability": probability,
            }
        )
    return probabilities, diagnostics


def process_video(args, model, processor, name: str) -> None:
    """Score every sliding window in one video with resumable checkpoints."""
    video_path = resolve_video_path(args.video_dir, name)
    stem = output_stem(name)
    output_path = args.output_dir / f"{stem}.json"
    detail_path = args.output_dir / "_calibration_details" / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    info = get_video_info(video_path)
    windows = list(iter_video_windows(info, args.frame_interval, args.window_seconds))
    expected = [str(window.center_frame) for window in windows]
    scores = load_json_dict(output_path) if args.resume else {}
    details = (
        load_json_dict(detail_path)
        if args.resume and args.save_calibration_details
        else {}
    )
    errors = load_json_dict(error_path) if args.resume else {}
    if (
        args.resume
        and is_complete(scores, expected)
        and (
            not args.save_calibration_details
            or set(details) == set(expected)
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
        if args.save_calibration_details:
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
                    not args.save_calibration_details
                    or key in details
                )
                and key not in errors
            ):
                continue
            try:
                forward_conversations = [
                    build_yesno_polarity_conversation(
                        video_path,
                        window,
                        args.sample_fps,
                        args.max_frames,
                        args.prompt,
                        threshold,
                        reverse=False,
                        precise_time=args.precise_time,
                    )
                    for threshold in CUMULATIVE_THRESHOLDS
                ]
                reverse_conversations = [
                    build_yesno_polarity_conversation(
                        video_path,
                        window,
                        args.sample_fps,
                        args.max_frames,
                        args.prompt,
                        threshold,
                        reverse=True,
                        precise_time=args.precise_time,
                    )
                    for threshold in CUMULATIVE_THRESHOLDS
                ]
                probabilities, threshold_details = (
                    polarity_calibrated_tail_probabilities(
                        model,
                        processor,
                        forward_conversations,
                        reverse_conversations,
                        yes_candidate=args.yes_candidate,
                        no_candidate=args.no_candidate,
                        temperature=args.likelihood_temperature,
                        optimized=args.optimized,
                        threshold_batch_size=args.threshold_batch_size,
                        prefix_cache=args.prefix_cache,
                    )
                )
                score, adjusted, violations = (
                    cumulative_score_from_tail_probabilities(
                        probabilities,
                        monotonic_projection=args.monotonic_projection,
                    )
                )
                scores[key] = score
                if args.save_calibration_details:
                    details[key] = {
                        "experiment": EXPERIMENT,
                        "thresholds": [
                            {
                                "threshold": threshold,
                                **detail,
                            }
                            for threshold, detail in zip(
                                CUMULATIVE_THRESHOLDS, threshold_details
                            )
                        ],
                        "monotonic_tail_probabilities": adjusted,
                        "monotonic_violations": violations,
                        "score": score,
                    }
                errors.pop(key, None)
                LOGGER.info(
                    "%s experiment=%s frame=%s time=%.3f-%.3f score=%.6f "
                    "tail_raw=%s tail_monotonic=%s violations=%d",
                    stem,
                    EXPERIMENT,
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
        "--save_calibration_details",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        args.model_path, args.device, args.attn_implementation
    )
    yes_id, no_id = validate_single_token_candidates(
        processor,
        args.yes_candidate,
        args.no_candidate,
    )
    LOGGER.info(
        "Experiment=%s YES=%r(token=%d) NO=%r(token=%d) optimized=%s",
        EXPERIMENT,
        args.yes_candidate,
        yes_id,
        args.no_candidate,
        no_id,
        args.optimized,
    )
    for name in names:
        try:
            process_video(args, model, processor, name)
        except Exception:
            LOGGER.exception("Failed to initialize/process video %s; continuing", name)


if __name__ == "__main__":
    main()
