import numpy as np

from covas_vad.evaluation import expand_clip_scores


def test_score_keys_are_sorted_numerically_before_expansion():
    expanded = expand_clip_scores(
        {"32": 0.3, "0": 0.1, "16": 0.2},
        frame_interval=2,
        smoothing_sigma=10.0,
        smooth=False,
    )
    assert np.array_equal(expanded, np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3]))

