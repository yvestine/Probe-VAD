"""Refine direct video anomaly scores using tentative video-level anomaly tags."""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.video_score_utils import (
    DEFAULT_MODEL_PATH,
    VideoWindow,
    atomic_write_json,
    build_video_conversation,
    get_video_info,
    is_complete,
    iter_video_windows,
    load_json_dict,
    load_model,
    load_video_names,
    output_stem,
    resolve_video_path,
    score_conversation,
    split_for_job,
)

LOGGER = logging.getLogger("video_refine_with_tag")
REFINE_PROMPT = (
    "You are a careful video anomaly detector for public surveillance scenes. "
    "The tentative activity phrases below were extracted from other high-scoring "
    "parts of the same video. They are hypotheses only: they may be inaccurate, "
    "irrelevant, or absent from the current segment. Do not raise the score merely "
    "because words such as assault, collision, theft, or fire occur in the prompt. "
    "Judge the current segment only from visible evidence. A score of 0 means "
    "ordinary safe activity and 1 means clearly abnormal, dangerous, violent, "
    "criminal, or emergency activity.\nTentative hypotheses: {phrases}"
)


def normalize_phrases(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts) if parts else "none"
    text = str(value or "").strip()
    if not text:
        return "none"
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    if isinstance(parsed, list):
        return ", ".join(str(item).strip() for item in parsed if str(item).strip()) or "none"
    return text


def phrase_for_video(phrases: dict, name: str) -> str:
    stem = output_stem(name)
    for key, value in phrases.items():
        if output_stem(key) == stem:
            return normalize_phrases(value)
    return "none"


def fuse_scores(initial: dict, refined: dict, alpha: float) -> dict:
    keys = sorted(set(initial) | set(refined), key=int)
    fused = {}
    for key in keys:
        if key in initial and key in refined:
            fused[key] = alpha * float(initial[key]) + (1.0 - alpha) * float(refined[key])
        elif key in initial:
            fused[key] = float(initial[key])
        else:
            fused[key] = float(refined[key])
    return fused


def process_video(args, model, processor, phrases: dict, name: str) -> None:
    video_path = resolve_video_path(args.video_dir, name)
    stem = output_stem(name)
    source_initial_path = args.initial_scores_dir / f"{stem}.json"
    initial = load_json_dict(source_initial_path)
    if not initial:
        raise FileNotFoundError(f"missing/empty initial scores: {source_initial_path}")

    initial_path = args.output_root / "initial" / f"{stem}.json"
    refined_path = args.output_root / "refined" / f"{stem}.json"
    fused_path = args.output_root / "fused" / f"{stem}.json"
    error_path = args.output_root / "_errors" / f"{stem}.json"
    # Snapshot the exact initial input so each ablation directory is standalone.
    atomic_write_json(initial_path, initial)

    info = get_video_info(video_path)
    windows = list(iter_video_windows(info, args.frame_interval, args.window_seconds))
    expected = [str(window.center_frame) for window in windows]
    missing_initial = sorted(set(expected) - set(initial), key=int)
    if missing_initial:
        raise RuntimeError(
            f"initial score file is incomplete ({len(missing_initial)} missing windows): "
            f"{source_initial_path}"
        )
    refined = load_json_dict(refined_path) if args.resume else {}
    errors = load_json_dict(error_path) if args.resume else {}

    if args.gate:
        gate_keys = {
            key for key in expected
            if args.gate_min <= float(initial[key]) <= args.gate_max
        }
        # The refined output remains a complete, evaluation-compatible time
        # series. Outside the uncertainty gate it is exactly the Initial score.
        for key in expected:
            if key not in gate_keys:
                refined[key] = float(initial[key])
                errors.pop(key, None)
        LOGGER.info(
            "Video %s clip-level gate [%.4f, %.4f]: refining %d/%d windows",
            stem, args.gate_min, args.gate_max, len(gate_keys), len(expected),
        )
        # Videos with zero gated clips are already complete after copying their
        # Initial sequence. Persist that sequence before the early-complete exit
        # so evaluation never silently drops those videos.
        atomic_write_json(refined_path, refined)
    else:
        gate_keys = set(expected)

    if args.resume and is_complete(refined, expected) and not errors:
        if args.save_fused:
            atomic_write_json(fused_path, fuse_scores(initial, refined, args.alpha))
        LOGGER.info("Skipping complete refined video %s", stem)
        return

    phrase_text = phrase_for_video(phrases, name)
    prompt = args.prompt_template.format(phrases=phrase_text)
    LOGGER.info("Video %s tentative hypotheses: %s", stem, phrase_text)
    for window in tqdm(windows, desc=f"{stem} refine", unit="window"):
        key = str(window.center_frame)
        if key not in gate_keys:
            continue
        if args.resume and key in refined and key not in errors:
            continue
        conversation = build_video_conversation(
            video_path,
            window,
            args.sample_fps,
            args.max_frames,
            prompt,
            args.score_mode,
        )
        try:
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
            refined[key] = score
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
        except Exception as exc:
            # A failed refinement should not destroy a valid Initial score.
            refined[key] = float(initial[key])
            errors[key] = {"raw_output": "", "error": repr(exc)}
            LOGGER.exception(
                "%s frame=%s time=%.3f-%.3f failed; retaining initial %.4f",
                stem, key, window.start_time, window.end_time, refined[key],
            )
        atomic_write_json(refined_path, refined)
        if args.save_fused:
            atomic_write_json(fused_path, fuse_scores(initial, refined, args.alpha))
        if errors:
            atomic_write_json(error_path, errors)
        elif error_path.exists():
            error_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--index_file", type=Path)
    parser.add_argument("--video_list", nargs="+")
    parser.add_argument("--initial_scores_dir", type=Path, required=True)
    parser.add_argument("--phrases_file", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument(
        "--gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only refine clips whose Initial score lies in the inclusive uncertainty gate.",
    )
    parser.add_argument("--gate_min", type=float, default=0.45)
    parser.add_argument("--gate_max", type=float, default=0.55)
    parser.add_argument(
        "--save_fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save an Initial/Refined weighted fusion in addition to Refined scores.",
    )
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument(
        "--score_mode",
        choices=["generated", "likelihood", "binary_likelihood"],
        default="generated",
        help=(
            "generated: discrete generation; likelihood: expected score over "
            "11 bracketed candidates; binary_likelihood: NORMAL/ANOMALOUS probability"
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
    parser.add_argument("--prompt_template", default=REFINE_PROMPT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if not 0 <= args.default_score <= 1:
        parser.error("--default_score must be between 0 and 1")
    if not 0 <= args.gate_min <= args.gate_max <= 1:
        parser.error("--gate_min and --gate_max must satisfy 0 <= min <= max <= 1")
    if args.likelihood_temperature <= 0:
        parser.error("--likelihood_temperature must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    with args.phrases_file.open() as handle:
        phrases = json.load(handle)
    names = split_for_job(
        load_video_names(args.video_dir, args.index_file, args.video_list),
        args.num_jobs,
        args.job_index,
    )
    model, processor = load_model(
        args.model_path, args.device, args.attn_implementation
    )
    for name in names:
        try:
            process_video(args, model, processor, phrases, name)
        except Exception:
            LOGGER.exception("Failed to initialize/process video %s; continuing", name)


if __name__ == "__main__":
    main()
