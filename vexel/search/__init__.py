"""vexel.search — VisualSearchEngine and SearchCandidate."""

from vexel.search.engine import SearchCandidate, VisualSearchEngine
from vexel.search.threshold import apply_score_gap_threshold

__all__ = ["VisualSearchEngine", "SearchCandidate", "apply_score_gap_threshold"]