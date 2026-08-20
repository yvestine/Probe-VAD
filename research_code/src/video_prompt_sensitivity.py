"""Multi-prompt cumulative scoring with one visual pass per video window.

The four prompt variants are evaluated together.  A window is decoded and
vision-encoded once; only the text suffixes (four prompts x ten thresholds)
are evaluated separately and written to the corresponding output directories.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from src.prompt_configs import get_prompt_spec
from src.run_manifest import build_manifest, ensure_manifest
from src.video_score_utils import (
    CUMULATIVE_THRESHOLDS,
    atomic_write_json,
    cumulative_score_from_tail_probabilities,
    get_video_info,
    is_complete,
    iter_video_windows,
    load_json_dict,
    load_video_names,
    output_stem,
    resolve_video_path,
    split_for_job,
)
from src.vlm_backends import load_backend

LOGGER = logging.getLogger("video_prompt_sensitivity")
DEFAULT_VARIANTS = (
    "visible_evidence",
    "reach_level",
    "no_less_than",
    "rated_above",
)
VARIANT_CHOICES = (
    "baseline",
    "visible_evidence",
    "reach_level",
    "no_less_than",
    "rated_above",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--index_file", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--prompt_file", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--question_template", default=None)
    parser.add_argument(
        "--prompt_variants",
        nargs="+",
        choices=VARIANT_CHOICES,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", choices=("auto", "videollama3", "qwen3_vl"), default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument(
        "--video_reader_backend",
        choices=("auto", "torchvision", "decord", "torchcodec"),
        default="auto",
    )
    parser.add_argument("--yes_candidate", default="YES")
    parser.add_argument("--no_candidate", default="NO")
    parser.add_argument("--length_normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--likelihood_temperature", type=float, default=1.0)
    parser.add_argument("--optimized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--threshold_batch_size",
        type=int,
        default=40,
        help="Text suffix batch size; all prompt variants still share one visual pass.",
    )
    parser.add_argument("--prefix_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monotonic_projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default_score", type=float, default=0.5)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--save_threshold_details", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--precise_time", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if len(set(args.prompt_variants)) != len(args.prompt_variants):
        parser.error("--prompt_variants must not contain duplicates")
    if args.likelihood_temperature <= 0:
        parser.error("--likelihood_temperature must be positive")
    if args.threshold_batch_size <= 0:
        parser.error("--threshold_batch_size must be positive")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint_interval must be positive")
    if not 0 <= args.default_score <= 1:
        parser.error("--default_score must be between 0 and 1")
    return args


def _load_states(args, prompt_specs, expected):
    states = {}
    for prompt_spec in prompt_specs:
        output_dir = args.output_root / prompt_spec.prompt_id
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_manifest(
            output_dir,
            build_manifest(args, prompt_spec, args.backend_name, args.backend_metadata),
        )
        states[prompt_spec.prompt_id] = {
            "output_dir": output_dir,
            "scores": {},
            "details": {},
            "errors": {},
            "dirty": 0,
        }
        for name in expected:
            stem = output_stem(name)
            state = states[prompt_spec.prompt_id]
            state.setdefault("files", {})[stem] = {
                "scores": load_json_dict(output_dir / f"{stem}.json") if args.resume else {},
                "details": load_json_dict(output_dir / "_threshold_details" / f"{stem}.json")
                if args.resume and args.save_threshold_details
                else {},
                "errors": load_json_dict(output_dir / "_errors" / f"{stem}.json") if args.resume else {},
            }
    return states


def _flush(args, states, stem):
    for state in states.values():
        file_state = state["files"][stem]
        atomic_write_json(state["output_dir"] / f"{stem}.json", file_state["scores"])
        if args.save_threshold_details:
            atomic_write_json(
                state["output_dir"] / "_threshold_details" / f"{stem}.json",
                file_state["details"],
            )
        error_path = state["output_dir"] / "_errors" / f"{stem}.json"
        if file_state["errors"]:
            atomic_write_json(error_path, file_state["errors"])
        elif error_path.exists():
            error_path.unlink()


def _process_video(args, backend, prompt_specs, states, name):
    video_path = resolve_video_path(args.video_dir, name)
    stem = output_stem(name)
    info = get_video_info(video_path)
    windows = list(iter_video_windows(info, args.frame_interval, args.window_seconds))
    LOGGER.info("Video %s: %d frames, %.3f FPS, %d windows", stem, info.frame_count, info.fps, len(windows))
    dirty = 0
    for window in tqdm(windows, desc=stem, unit="window"):
        key = str(window.center_frame)
        pending = []
        for prompt_spec in prompt_specs:
            file_state = states[prompt_spec.prompt_id]["files"][stem]
            complete = (
                args.resume
                and key in file_state["scores"]
                and key not in file_state["errors"]
                and (not args.save_threshold_details or key in file_state["details"])
            )
            if not complete:
                pending.append(prompt_spec)
        if not pending:
            continue

        try:
            results = backend.score_prompt_variants(video_path, window, prompt_specs, args)
            for prompt_spec in pending:
                probabilities, backend_details = results[prompt_spec.prompt_id]
                score, adjusted, violations = cumulative_score_from_tail_probabilities(
                    probabilities, monotonic_projection=args.monotonic_projection
                )
                file_state = states[prompt_spec.prompt_id]["files"][stem]
                file_state["scores"][key] = score
                file_state["errors"].pop(key, None)
                if args.save_threshold_details:
                    file_state["details"][key] = {
                        "thresholds": [
                            {"threshold": threshold, **detail}
                            for threshold, detail in zip(CUMULATIVE_THRESHOLDS, backend_details)
                        ],
                        "monotonic_tail_probabilities": adjusted,
                        "monotonic_violations": violations,
                        "score": score,
                    }
        except Exception as exc:
            LOGGER.exception("%s frame=%s failed for combined prompt batch", stem, key)
            for prompt_spec in pending:
                file_state = states[prompt_spec.prompt_id]["files"][stem]
                file_state["scores"][key] = args.default_score
                file_state["errors"][key] = {"raw_output": "", "error": repr(exc)}
        dirty += 1
        if dirty >= args.checkpoint_interval:
            _flush(args, states, stem)
            dirty = 0
    if dirty:
        _flush(args, states, stem)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    prompt_specs = [
        get_prompt_spec(
            variant,
            args.prompt_file,
            args.prompt,
            args.question_template,
        )
        for variant in args.prompt_variants
    ]
    names = split_for_job(
        load_video_names(args.video_dir, args.index_file, None),
        args.num_jobs,
        args.job_index,
    )
    args.backend_name = args.backend
    args.backend_metadata = {}
    backend = load_backend(
        args.model_path,
        args.device,
        args.attn_implementation,
        args.backend,
        args.video_reader_backend,
    )
    args.backend_name = backend.name
    args.backend_metadata = backend.metadata()
    states = _load_states(args, prompt_specs, names)
    for name in names:
        try:
            _process_video(args, backend, prompt_specs, states, name)
        except Exception:
            LOGGER.exception("Failed to initialize/process video %s; continuing", name)


if __name__ == "__main__":
    main()
