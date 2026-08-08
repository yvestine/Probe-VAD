from pathlib import Path

import pytest

from covas_vad.utils import (
    VideoInfo,
    is_complete,
    iter_video_windows,
    output_stem,
)


def test_window_centers_and_boundaries():
    info = VideoInfo(fps=10.0, frame_count=101, duration=10.1)
    windows = list(iter_video_windows(info, frame_interval=20, window_seconds=4.0))
    assert [window.center_frame for window in windows] == [0, 20, 40, 60, 80, 100]
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == pytest.approx(2.0)
    assert windows[-1].start_time == pytest.approx(8.0)
    assert windows[-1].end_time == pytest.approx(10.1)


def test_output_stem_preserves_dataset_identifiers_with_dots():
    name = "Spectre.2015__#01-08-58_01-09-20_label_B1-B2-0"
    assert output_stem(name) == name
    assert output_stem(f"{name}.mp4") == name
    assert output_stem(str(Path("nested") / f"{name}.avi")) == name


def test_complete_score_mapping_requires_exact_numeric_values():
    expected = ["0", "16"]
    assert is_complete({"0": 0.1, "16": 0.2}, expected)
    assert not is_complete({"0": 0.1}, expected)
    assert not is_complete({"0": 0.1, "16": "0.2"}, expected)

