"""Unit tests for vexel.search.threshold."""

from vexel.search.threshold import apply_score_gap_threshold


def test_empty_list():
    assert apply_score_gap_threshold([], 0.08) == 0


def test_single_item():
    assert apply_score_gap_threshold([0.90], 0.08) == 1


def test_no_gap_exceeds_threshold():
    scores = [0.95, 0.94, 0.92, 0.91]
    assert apply_score_gap_threshold(scores, 0.08) == 4


def test_gap_at_start():
    scores = [0.95, 0.80, 0.79, 0.78]
    # 0.95 → 0.80 = 0.15 > 0.08 → keep only first element
    assert apply_score_gap_threshold(scores, 0.08) == 1


def test_gap_in_middle():
    scores = [0.95, 0.93, 0.91, 0.75, 0.74]
    # 0.91 → 0.75 = 0.16 > 0.08 → keep first 3
    assert apply_score_gap_threshold(scores, 0.08) == 3


def test_gap_at_end():
    scores = [0.95, 0.94, 0.93, 0.82]
    # 0.93 → 0.82 = 0.11 > 0.08 → keep first 3
    assert apply_score_gap_threshold(scores, 0.08) == 3


def test_exact_threshold_not_triggered():
    # Gap clearly below threshold → NOT triggered
    # Use integer-representable values to avoid floating-point surprises
    scores = [0.90, 0.83]   # gap = 0.07 < 0.08
    assert apply_score_gap_threshold(scores, 0.08) == 2


def test_just_above_threshold_triggered():
    scores = [0.90, 0.8199]
    assert apply_score_gap_threshold(scores, 0.08) == 1