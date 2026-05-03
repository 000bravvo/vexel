"""
Score-gap thresholding for visual search result pruning.

Principle:
    After ANN retrieval the ranked result list may contain a sharp relevance
    cliff — a large score drop between two consecutive hits.  Everything below
    that cliff is noise from the HNSW graph and should be discarded.

    We scan the sorted list and drop every result **after** the first gap that
    exceeds `gap_threshold`.  This produces a shorter but higher-precision set
    without requiring a fixed score cut-off.
"""

from __future__ import annotations


def apply_score_gap_threshold(
    scores: list[float],
    gap_threshold: float,
) -> int:
    """
    Return the number of results to keep based on the first large score gap.

    Args:
        scores:        Cosine similarity scores, **descending** order.
        gap_threshold: Maximum allowed gap between consecutive scores.
                       First gap larger than this truncates the list.

    Returns:
        ``keep_count`` — slice the original list to ``results[:keep_count]``.

    Examples::

        >>> apply_score_gap_threshold([0.95, 0.93, 0.91, 0.75, 0.74], 0.08)
        3  # gap 0.91→0.75 = 0.16 > 0.08 → keep first 3

        >>> apply_score_gap_threshold([0.95, 0.93, 0.91], 0.08)
        3  # no large gap → keep all
    """
    if not scores:
        return 0

    for i in range(1, len(scores)):
        if (scores[i - 1] - scores[i]) > gap_threshold:
            return i

    return len(scores)