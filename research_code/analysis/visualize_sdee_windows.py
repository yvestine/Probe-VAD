"""Generate CPU-only E5-SDEE S1/M1...S5/M5 inspection panels."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from src.sdee_motion import (
    construct_sdee_sequence,
    decode_window_cv2,
    visualization_panel,
)
from src.video_score_utils import (
    atomic_write_json,
    get_video_info,
    iter_video_windows,
    load_video_names,
    output_stem,
    resolve_video_path,
)


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
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cell_width", type=int, default=384)
    args = parser.parse_args()
    if args.limit <= 0 or args.motion_fps <= 0 or args.cell_width <= 0:
        parser.error("limit, motion FPS and cell width must be positive")
    if args.temporal_bins != 5:
        parser.error("E5-SDEE requires exactly five bins")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = load_video_names(args.video_dir, args.index_file, args.video_list)
    candidates = []
    for name in tqdm(names, desc="index windows", unit="video"):
        video_path = resolve_video_path(args.video_dir, name)
        info = get_video_info(video_path)
        candidates.extend(
            (name, window)
            for window in iter_video_windows(
                info,
                args.frame_interval,
                args.window_seconds,
            )
        )
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: min(args.limit, len(candidates))]
    manifest = {}
    failures = {}
    for name, window in tqdm(selected, desc="render SDEE", unit="window"):
        stem = output_stem(name)
        key = f"{stem}__f{window.center_frame}"
        video_path = resolve_video_path(args.video_dir, name)
        try:
            frames, timestamps = decode_window_cv2(
                video_path,
                window.start_time,
                window.end_time,
                args.motion_fps,
            )
            sequence, sequence_times, details = construct_sdee_sequence(
                frames,
                timestamps,
                window.start_time,
                window.end_time,
                temporal_bins=args.temporal_bins,
                align_global_motion=args.motion_mode == "aligned",
                shuffle_motion=args.shuffle_motion,
            )
            panel_path = args.output_dir / f"{key}.png"
            visualization_panel(sequence, args.cell_width).save(panel_path)
            manifest[key] = {
                "video": str(video_path),
                "center_frame": window.center_frame,
                "window": [window.start_time, window.end_time],
                "panel": str(panel_path),
                "sequence_timestamps": sequence_times,
                **details,
            }
        except Exception as exc:
            failures[key] = {
                "video": str(video_path),
                "center_frame": window.center_frame,
                "error": repr(exc),
            }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    if failures:
        atomic_write_json(args.output_dir / "errors.json", failures)
    print(
        f"Rendered {len(manifest)}/{len(selected)} windows to {args.output_dir}; "
        f"failures={len(failures)}"
    )


if __name__ == "__main__":
    main()
