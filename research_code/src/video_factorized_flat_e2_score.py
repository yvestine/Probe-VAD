"""E2 A/B option-swap calibration with flat exact-class queries (isolated).

Preserves E2 likelihood engines, video handling and checkpoint/resume strategy.
Only question semantics, the 11-class grid and final aggregation differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

from tqdm import tqdm
from src import video_ab_calibrated_score as e2
from src import video_score_utils as u

EXPERIMENT = "factorized_flat_e2_11class"
CANDIDATE_SCORES = tuple(i / 10 for i in range(11))
LOGGER = logging.getLogger(EXPERIMENT)


def build_ab_threshold_conversation(video_path, window, sample_fps, max_frames,
                                    prompt, threshold, *, swapped,
                                    a_candidate="A", b_candidate="B", precise_time=False):
    """Same signature/video metadata as E2; threshold now means a flat score."""
    conversation = e2.build_ab_threshold_conversation(
        video_path, window, sample_fps, max_frames, prompt, threshold,
        swapped=swapped, a_candidate=a_candidate, b_candidate=b_candidate,
        precise_time=precise_time)
    positive = f"[{threshold:.1f}] is the single most appropriate anomaly severity score."
    negative = f"[{threshold:.1f}] is not the single most appropriate anomaly severity score."
    a_meaning, b_meaning = (negative, positive) if swapped else (positive, negative)
    conversation[1]["content"][1]["text"] = (
        f"Is [{threshold:.1f}] the single most appropriate anomaly severity score "
        "for this video segment, based only on visible evidence?\n"
        f"{a_candidate}: {a_meaning}\n{b_candidate}: {b_meaning}\n"
        f"Answer exactly {a_candidate} or {b_candidate}."
    )
    return conversation


def aggregate_calibrated_logits(raw_logits, temperature=1.0):
    if temperature != 1.0:
        raise ValueError("This experiment fixes T=1.0")
    if len(raw_logits) != 22:
        raise ValueError("Expected 11 forward and 11 swapped likelihood pairs")
    details = []
    for candidate, forward, swapped in zip(CANDIDATE_SCORES, raw_logits[:11], raw_logits[11:]):
        values = [forward["a_logit"], forward["b_logit"], swapped["a_logit"], swapped["b_logit"]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite A/B likelihood")
        # Reuse E2's orientation and margin averaging verbatim. Discard sigmoid.
        _, margin, margin_2 = e2.calibrated_margin_probability(
            *values[:2], swapped_a_logit=values[2], swapped_b_logit=values[3],
            temperature=temperature)
        details.append(dict(candidate=candidate, forward_a_logit=values[0],
            forward_b_logit=values[1], swapped_a_logit=values[2], swapped_b_logit=values[3],
            margin_1=values[0]-values[1], margin_2=margin_2, calibrated_margin=margin))
    peak = max(d["calibrated_margin"] for d in details)
    masses = [math.exp(d["calibrated_margin"]-peak) for d in details]
    total = sum(masses)
    for d, mass in zip(details, masses):
        d["class_weight"] = mass / total
    score = sum(d["candidate"] * d["class_weight"] for d in details)
    return score, details


def option_calibrated_flat_score(model, processor, forward_conversations, swapped_conversations,
                                 *, a_candidate="A", b_candidate="B", temperature=1.0,
                                 optimized=True, threshold_batch_size=10, prefix_cache=True):
    if len(forward_conversations) != 11 or len(swapped_conversations) != 11:
        raise ValueError("Both orientations must contain eleven questions")
    conversations = list(forward_conversations) + list(swapped_conversations)
    if optimized:
        raw = e2._candidate_logits_optimized(model, processor, conversations,
            a_candidate, b_candidate, threshold_batch_size, prefix_cache)
    else:
        raw = e2._candidate_logits_reference(model, processor, conversations, a_candidate, b_candidate)
    return aggregate_calibrated_logits(raw, temperature)


def read_json(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def valid_window(score, detail):
    try:
        if not math.isfinite(score) or not 0 <= score <= 1 or detail["score"] != score:
            return False
        ds = detail["candidates"]
        if len(ds) != 11:
            return False
        raw = [dict(a_logit=d["forward_a_logit"], b_logit=d["forward_b_logit"]) for d in ds]
        raw += [dict(a_logit=d["swapped_a_logit"], b_logit=d["swapped_b_logit"]) for d in ds]
        expected_score, expected = aggregate_calibrated_logits(raw)
        return abs(expected_score-score) < 1e-10 and all(
            math.isfinite(d[k]) and abs(d[k]-ref[k]) < 1e-10
            for d, ref in zip(ds, expected)
            for k in ("candidate", "margin_1", "margin_2", "calibrated_margin", "class_weight"))
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def process_video(args, model, processor, name):
    video_path = u.resolve_video_path(args.video_dir, name)
    stem = u.output_stem(name)
    output_path = args.output_dir / f"{stem}.json"
    detail_path = args.output_dir / "_calibration_details" / f"{stem}.json"
    error_path = args.output_dir / "_errors" / f"{stem}.json"
    windows = list(u.iter_video_windows(u.get_video_info(video_path), args.frame_interval, args.window_seconds))
    scores, details, errors = read_json(output_path), read_json(detail_path), read_json(error_path)
    dirty = int("__video__" in errors)
    errors.pop("__video__", None)

    def checkpoint():
        nonlocal dirty
        if dirty:
            # Original evaluator uses JSON insertion order; retain numerical order on resume.
            u.atomic_write_json(detail_path, {k: details[k] for k in sorted(details, key=int)})
            u.atomic_write_json(output_path, {k: scores[k] for k in sorted(scores, key=int)})
            u.atomic_write_json(error_path, errors)
            dirty = 0

    try:
        for window in tqdm(windows, desc=stem, unit="window"):
            key = str(window.center_frame)
            if key not in errors and valid_window(scores.get(key), details.get(key, {})):
                continue
            try:
                prompts = [[build_ab_threshold_conversation(video_path, window,
                    args.sample_fps, args.max_frames, args.prompt, candidate,
                    swapped=swapped, a_candidate=args.a_candidate, b_candidate=args.b_candidate,
                    precise_time=args.precise_time) for candidate in CANDIDATE_SCORES]
                    for swapped in (False, True)]
                score, ds = option_calibrated_flat_score(model, processor, *prompts,
                    a_candidate=args.a_candidate, b_candidate=args.b_candidate,
                    temperature=args.likelihood_temperature, optimized=args.optimized,
                    threshold_batch_size=args.threshold_batch_size, prefix_cache=args.prefix_cache)
                scores[key] = score
                details[key] = dict(experiment=EXPERIMENT, candidates=ds, score=score,
                    start_time=window.start_time, end_time=window.end_time)
                errors.pop(key, None)
            except Exception as exc:
                # Keep failures explicit; never evaluate fallback-filled windows.
                scores.pop(key, None)
                details.pop(key, None)
                errors[key] = {"error": repr(exc)}
                LOGGER.exception("%s frame=%s failed", stem, key)
            dirty += 1
            if dirty >= args.checkpoint_interval:
                checkpoint()
    finally:
        checkpoint()
    return len(errors)


def validate_outputs(args, names):
    missing = failures = windows_total = 0
    for name in names:
        stem = u.output_stem(name)
        scores = read_json(args.output_dir / f"{stem}.json")
        details = read_json(args.output_dir / "_calibration_details" / f"{stem}.json")
        errors = read_json(args.output_dir / "_errors" / f"{stem}.json")
        windows = list(u.iter_video_windows(u.get_video_info(u.resolve_video_path(args.video_dir, name)),
                                           args.frame_interval, args.window_seconds))
        keys = [str(w.center_frame) for w in windows]
        windows_total += len(keys)
        failures += len(errors)
        missing += sum(not valid_window(scores.get(k), details.get(k, {})) for k in keys)
        if set(scores)-set(keys) or set(details)-set(keys):
            raise ValueError(f"Unexpected window keys in {stem}")
        # Fail before legacy evaluator can pad, skip, or misorder scores.
        if list(scores) != [k for k in keys if k in scores]:
            raise ValueError(f"Score ordering mismatch in {stem}")
    report = dict(videos=len(names), expected_windows=windows_total,
                  invalid_or_missing_windows=missing, failure_records=failures,
                  complete=not (missing or failures))
    u.atomic_write_json(args.output_dir / "_coverage.json", report)
    if not report["complete"]:
        raise RuntimeError(f"Incomplete experiment: {report}")
    return report


def manifest_for(args, names):
    return dict(experiment=EXPERIMENT, schema=1, names=names,
        index_sha256=hashlib.sha256(args.index_file.read_bytes()).hexdigest(),
        model=args.model_path, video_dir=str(args.video_dir.resolve()), prompt=args.prompt,
        a=args.a_candidate, b=args.b_candidate, temperature=args.likelihood_temperature,
        frame_interval=args.frame_interval, window_seconds=args.window_seconds,
        sample_fps=args.sample_fps, max_frames=args.max_frames, precise_time=args.precise_time,
        optimized=args.optimized, prefix_cache=args.prefix_cache,
        attention=args.attn_implementation, smoke=args.smoke,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(e2.__file__).read_bytes()+Path(u.__file__).read_bytes()).hexdigest())


def guard_output(path, manifest, validate_only=False):
    marker = path / "_experiment.json"
    if marker.exists():
        if read_json(marker) != manifest:
            raise RuntimeError("Configuration differs; use a NEW independent results directory")
    else:
        if validate_only or (path.exists() and any(path.iterdir())):
            raise RuntimeError("Refusing unowned/nonempty output directory")
        path.mkdir(parents=True, exist_ok=True)
        u.atomic_write_json(marker, manifest)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("video_dir", "index_file", "output_dir"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--model_path", default=u.DEFAULT_MODEL_PATH)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--threshold_batch_size", type=int, default=10)
    p.add_argument("--checkpoint_interval", type=int, default=20)
    p.add_argument("--optimized", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefix_cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true", help="First two COMPLETE videos of the full index")
    p.add_argument("--validate_only", action="store_true")
    args = p.parse_args()
    if min(args.threshold_batch_size, args.checkpoint_interval) <= 0:
        p.error("Batch/checkpoint sizes must be positive")
    # Fixed matched MSAD protocol; no test-set tuning switches.
    args.frame_interval, args.window_seconds, args.sample_fps, args.max_frames = 16, 10., 2., 10
    args.precise_time, args.likelihood_temperature = True, 1.
    args.prompt, args.a_candidate, args.b_candidate = e2.CUMULATIVE_PROMPT, "A", "B"
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    names = u.load_video_names(args.video_dir, args.index_file)
    if len(names) != 360 or len({u.output_stem(n) for n in names}) != 360:
        raise ValueError("Require complete unique 360-video index")
    if args.smoke:
        names = names[:2]
    guard_output(args.output_dir, manifest_for(args, names), args.validate_only)
    if not args.validate_only:
        import fcntl
        with (args.output_dir / "_run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            model, processor = u.load_model(args.model_path, args.device, args.attn_implementation)
            a_id, b_id = e2.validate_single_token_options(processor, args.a_candidate, args.b_candidate)
            u.atomic_write_json(args.output_dir / "_tokenizer.json", {"A": a_id, "B": b_id})
            for name in names:
                try:
                    process_video(args, model, processor, name)
                except Exception as exc:
                    u.atomic_write_json(args.output_dir / "_errors" / f"{u.output_stem(name)}.json",
                                        {"__video__": {"error": repr(exc)}})
                    LOGGER.exception("Failed video %s", name)
            print(json.dumps(validate_outputs(args, names)))
    else:
        print(json.dumps(validate_outputs(args, names)))

if __name__ == "__main__":
    main()
