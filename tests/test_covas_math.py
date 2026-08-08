import pytest

from covas_vad.utils import (
    cumulative_score_from_tail_probabilities,
    decreasing_isotonic_projection,
)


def test_pava_projects_to_non_increasing_sequence():
    projected = decreasing_isotonic_projection([0.9, 0.5, 0.7, 0.2])
    assert projected == pytest.approx([0.9, 0.6, 0.6, 0.2])
    assert all(left >= right for left, right in zip(projected, projected[1:]))


def test_cumulative_score_is_tail_probability_mean():
    raw = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    score, adjusted, violations = cumulative_score_from_tail_probabilities(raw)
    assert score == pytest.approx(0.55)
    assert adjusted == pytest.approx(raw)
    assert violations == 0


def test_cumulative_score_clips_and_projects():
    score, adjusted, violations = cumulative_score_from_tail_probabilities(
        [1.2, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -0.2]
    )
    assert adjusted == pytest.approx(
        [1.0, 0.75, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    )
    assert score == pytest.approx(sum(adjusted) / 10)
    assert violations == 1
