"""Direct video scoring with A/B option-swap calibrated threshold likelihoods.

This is an isolated ablation entry point.  It reuses the existing cumulative
likelihood implementation without changing the E0 YES/NO scorer:

* E1: A means severity >= threshold; B means severity < threshold.
* E2: evaluate both that assignment and its A/B-swapped counterpart, average
  the two anomaly-oriented margins, then apply sigmoid.

Both modes observe raw video clips only.  They do not generate captions,
anomaly tags, refinements, or fused scores.
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

LOGGER = logging.getLogger("video_ab_calibrated_score")


def stable_sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid."""
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def calibrated_margin_probability(
    forward_a_logit: float,
    forward_b_logit: float,
    *,
    swapped_a_logit: float | None = None,
    swapped_b_logit: float | None = None,
    temperature: float = 1.0,
) -> Tuple[float, float, float | None]:
    """Return sigmoid of the anomaly-oriented E1 or E2 option margin.

    In the forward prompt, A is anomalous and B is normal:
    ``m1 = logit(A) - logit(B)``.

    In the swapped prompt, B is anomalous and A is normal:
    ``m2 = logit(B) - logit(A)``.

    E2 deliberately averages margins before applying sigmoid.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if (swapped_a_logit is None) != (swapped_b_logit is None):
        raise ValueError("both swapped logits must be provided together")

    margin_1 = float(forward_a_logit) - float(forward_b_logit)
    margin_2 = None
    calibrated_margin = margin_1
    if swapped_a_logit is not None and swapped_b_logit is not None:
        margin_2 = float(swapped_b_logit) - float(swapped_a_logit)
        calibrated_margin = (margin_1 + margin_2) / 2.0

    probability = stable_sigmoid(calibrated_margin / temperature)
    return probability, calibrated_margin, margin_2


def build_ab_threshold_conversation(
    video_path: Path,
    window: VideoWindow,
    sample_fps: float,
    max_frames: int,
    prompt: str,
    threshold: float,
    *,
    swapped: bool,
    a_candidate: str = "A",
    b_candidate: str = "B",
    precise_time: bool = False,
) -> List[Dict[str, Any]]:
    """Build one A/B threshold prompt while keeping option order fixed."""
    at_least = f"The anomaly severity is at least {threshold:.1f}."
    below = f"The anomaly severity is strictly below {threshold:.1f}."
    a_meaning, b_meaning = (below, at_least) if swapped else (at_least, below)
    question = (
        "Choose exactly one option based only on visible evidence in the "
        "current video segment.\n"
        f"{a_candidate}: {a_meaning}\n"
        f"{b_candidate}: {b_meaning}\n"
        f"Answer exactly {a_candidate} or {b_candidate}."
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


def validate_single_token_options(
    processor, a_candidate: str = "A", b_candidate: str = "B"
) -> Tuple[int, int]:
    """Validate the exact continuation tokens used by the optimized scorer."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor does not expose a tokenizer")
    a_ids = tokenizer(a_candidate, add_special_tokens=False).input_ids
    b_ids = tokenizer(b_candidate, add_special_tokens=False).input_ids
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise ValueError(
            "A/B logit scoring requires one-token candidates; "
            f"{a_candidate!r}={a_ids}, {b_candidate!r}={b_ids}"
        )
    if a_ids[0] == b_ids[0]:
        raise ValueError("A and B candidates must map to different token IDs")
    return int(a_ids[0]), int(b_ids[0])


def _candidate_logits_reference(
    model,
    processor,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    a_candidate: str,
    b_candidate: str,
) -> List[Dict[str, float]]:
    """Slow per-prompt reference implementation for numerical validation."""
    diagnostics: List[Dict[str, float]] = []
    for conversation in conversations:
        _, a_logit, b_logit = threshold_yes_no_likelihood(
            model,
            processor,
            conversation,
            yes_candidate=a_candidate,
            no_candidate=b_candidate,
            length_normalize=False,
            temperature=1.0,
        )
        diagnostics.append(
            {
                "a_logit": float(a_logit),
                "b_logit": float(b_logit),
            }
        )
    return diagnostics


def _candidate_logits_optimized(
    model,
    processor,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    a_candidate: str,
    b_candidate: str,
    threshold_batch_size: int,
    prefix_cache: bool,
) -> List[Dict[str, float]]:
    """Evaluate all E1/E2 prompts with one shared visual encoding."""
    _, raw_details = cumulative_threshold_likelihood_optimized(
        model,
        processor,
        conversations,
        yes_candidate=a_candidate,
        no_candidate=b_candidate,
        temperature=1.0,
        threshold_batch_size=threshold_batch_size,
        prefix_cache=prefix_cache,
    )
    return [
        {
            # The reused utility reports log-softmax values.  Their difference
            # is exactly the corresponding raw-logit difference.
            "a_logit": float(detail["yes_loglik"]),
            "b_logit": float(detail["no_loglik"]),
        }
        for detail in raw_details
    ]


def option_calibrated_tail_probabilities(
    model,
    processor,
    forward_conversations: Sequence[Sequence[Mapping[str, Any]]],
    swapped_conversations: Sequence[Sequence[Mapping[str, Any]]] | None,
    *,
    a_candidate: str,
    b_candidate: str,
    temperature: float,
    optimized: bool,
    threshold_batch_size: int,
    prefix_cache: bool,
) -> Tuple[List[float], List[Dict[str, float | None]]]:
    """Compute E1 or E2 probabilities and per-threshold diagnostics."""
    if len(forward_conversations) != len(CUMULATIVE_THRESHOLDS):
        raise ValueError(
            f"expected {len(CUMULATIVE_THRESHOLDS)} forward prompts, "
            f"got {len(forward_conversations)}"
        )
    if swapped_conversations is not None and (
        len(swapped_conversations) != len(forward_conversations)
    ):
        raise ValueError("forward and swapped prompt counts must match")

    all_conversations = list(forward_conversations)
    if swapped_conversations is not None:
        all_conversations.extend(swapped_conversations)

    if optimized:
        raw_logits = _candidate_logits_optimized(
            model,
            processor,
            all_conversations,
            a_candidate,
            b_candidate,
            threshold_batch_size,
            prefix_cache,
        )
    else:
        raw_logits = _candidate_logits_reference(
            model,
            processor,
            all_conversations,
            a_candidate,
            b_candidate,
        )

    split = len(forward_conversations)
    forward_logits = raw_logits[:split]
    swapped_logits = raw_logits[split:] if swapped_conversations is not None else None
    probabilities: List[float] = []
    diagnostics: List[Dict[str, float | None]] = []

    for index, forward in enumerate(forward_logits):
        swapped = swapped_logits[index] if swapped_logits is not None else None
        probability, margin, margin_2 = calibrated_margin_probability(
            forward["a_logit"],
            forward["b_logit"],
            swapped_a_logit=None if swapped is None else swapped["a_logit"],
            swapped_b_logit=None if swapped is None else swapped["b_logit"],
            temperature=temperature,
        )
        margin_1 = forward["a_logit"] - forward["b_logit"]
        probabilities.append(probability)
        diagnostics.append(
            {
                "forward_a_logit": forward["a_logit"],
                "forward_b_logit": forward["b_logit"],
                "swapped_a_logit": None if swapped is None else swapped["a_logit"],
                "swapped_b_logit": None if swapped is None else swapped["b_logit"],
                "margin_1": margin_1,
                "margin_2": margin_2,
                "calibrated_margin": margin,
                "tail_probability": probability,
            }
        )
    return probabilities, diagnostics


def process_video(args, model, processor, name: str) -> None:
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
        args.experiment,
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
                    build_ab_threshold_conversation(
                        video_path,
                        window,
                        args.sample_fps,
                        args.max_frames,
                        args.prompt,
                        threshold,
                        swapped=False,
                        a_candidate=args.a_candidate,
                        b_candidate=args.b_candidate,
                        precise_time=args.precise_time,
                    )
                    for threshold in CUMULATIVE_THRESHOLDS
                ]
                swapped_conversations = (
                    [
                        build_ab_threshold_conversation(
                            video_path,
                            window,
                            args.sample_fps,
                            args.max_frames,
                            args.prompt,
                            threshold,
                            swapped=True,
                            a_candidate=args.a_candidate,
                            b_candidate=args.b_candidate,
                            precise_time=args.precise_time,
                        )
                        for threshold in CUMULATIVE_THRESHOLDS
                    ]
                    if args.experiment == "e2"
                    else None
                )
                probabilities, threshold_details = (
                    option_calibrated_tail_probabilities(
                        model,
                        processor,
                        forward_conversations,
                        swapped_conversations,
                        a_candidate=args.a_candidate,
                        b_candidate=args.b_candidate,
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
                        "experiment": args.experiment,
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
                    args.experiment,
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
                    args.experiment,
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
    parser.add_argument("--experiment", choices=("e1", "e2"), default="e2")
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
    parser.add_argument("--a_candidate", default="A")
    parser.add_argument("--b_candidate", default="B")
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
    if args.a_candidate == args.b_candidate:
        parser.error("--a_candidate and --b_candidate must differ")
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
    a_id, b_id = validate_single_token_options(
        processor, args.a_candidate, args.b_candidate
    )
    LOGGER.info(
        "Experiment=%s A=%r(token=%d) B=%r(token=%d) optimized=%s",
        args.experiment,
        args.a_candidate,
        a_id,
        args.b_candidate,
        b_id,
        args.optimized,
    )
    for name in names:
        try:
            process_video(args, model, processor, name)
        except Exception:
            LOGGER.exception("Failed to initialize/process video %s; continuing", name)


if __name__ == "__main__":
    main()
