"""Validate COVAS-VAD video and annotation inputs without using a GPU."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from covas_vad.utils import (
    get_video_info,
    iter_video_windows,
    output_stem,
    resolve_video_path,
)


def normalized_name(value: str) -> str:
    path = Path(value)
    return path.stem if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--index_file", type=Path, required=True)
    parser.add_argument("--temporal_annotation_file", type=Path)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--window_seconds", type=float, default=10.0)
    args = parser.parse_args()

    if not args.video_dir.is_dir():
        parser.error(f"video directory does not exist: {args.video_dir}")
    if not args.index_file.is_file():
        parser.error(f"index file does not exist: {args.index_file}")

    rows = [
        line.split()
        for line in args.index_file.read_text().splitlines()
        if line.strip()
    ]
    problems: list[str] = []
    stem_to_names: dict[str, list[str]] = defaultdict(list)
    resolved_names: set[str] = set()
    total_windows = 0

    for line_number, row in enumerate(rows, start=1):
        if len(row) < 4:
            problems.append(
                f"index line {line_number}: expected at least 4 fields, got {len(row)}"
            )
            continue
        name = row[0]
        stem_to_names[output_stem(name)].append(name)
        resolved_names.add(normalized_name(name))
        try:
            path = resolve_video_path(args.video_dir, name)
            info = get_video_info(path)
            total_windows += sum(
                1
                for _ in iter_video_windows(
                    info, args.frame_interval, args.window_seconds
                )
            )
        except Exception as exc:
            problems.append(f"{name}: {exc}")

    for stem, names in stem_to_names.items():
        if len(names) > 1:
            problems.append(
                f"duplicate output stem {stem!r} would overwrite: {names}"
            )

    if args.temporal_annotation_file:
        if not args.temporal_annotation_file.is_file():
            problems.append(
                f"temporal annotation file does not exist: "
                f"{args.temporal_annotation_file}"
            )
        else:
            temporal_names = {
                normalized_name(line.split()[0])
                for line in args.temporal_annotation_file.read_text().splitlines()
                if line.strip()
            }
            missing = sorted(resolved_names - temporal_names)
            if missing:
                preview = ", ".join(missing[:10])
                suffix = " ..." if len(missing) > 10 else ""
                problems.append(
                    f"{len(missing)} index videos have no temporal annotation: "
                    f"{preview}{suffix}"
                )

    print(f"index videos: {len(rows)}")
    print(f"estimated clips: {total_windows}")
    print(f"problems: {len(problems)}")
    for problem in problems:
        print(f"ERROR: {problem}")
    if problems:
        sys.exit(1)
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()

