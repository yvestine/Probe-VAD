"""Patch the cached VideoLLaMA3 loader for ffmpeg frame/timestamp drift.

The remote processor builds timestamps from the requested duration, while
ffmpeg can return a few extra decoded frames.  Its subsequent subsampling then
indexes past the timestamp array.  This idempotent patch aligns timestamps
before that indexing.  It is run by the MSAD recovery entry point only.
"""

from __future__ import annotations

from pathlib import Path


MARKER = "# URF-HVAA timestamp alignment patch"
NEEDLE = "        if max_frames is not None and len(frames) > max_frames:\n"
INSERT = (
    f"        {MARKER}\n"
    "        if len(timestamps) < len(frames):\n"
    "            timestamps = np.linspace(start_time, start_time + duration, len(frames))\n"
    "\n"
)


def main() -> None:
    roots = [
        Path.home() / ".cache/huggingface/modules/transformers_modules",
        Path("/workspace/gujiawei/.cache/huggingface/modules/transformers_modules"),
    ]
    candidates = [
        path
        for root in roots
        if root.exists()
        for path in root.glob("**/processing_videollama3.py")
    ]
    if not candidates:
        raise SystemExit("VideoLLaMA3 processing_videollama3.py was not found")
    changed = 0
    for path in candidates:
        text = path.read_text()
        if MARKER in text:
            continue
        if NEEDLE not in text:
            raise SystemExit(f"unexpected VideoLLaMA3 loader layout: {path}")
        path.write_text(text.replace(NEEDLE, INSERT + NEEDLE, 1))
        changed += 1
        print(f"patched {path}")
    print(f"VideoLLaMA3 timestamp patch ready ({changed} file(s) changed)")


if __name__ == "__main__":
    main()
