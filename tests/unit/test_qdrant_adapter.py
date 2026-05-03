"""Unit tests for QdrantAdapter._build_filter."""

from vexel.store.qdrant import QdrantAdapter
from vexel.config import QdrantStoreConfig


def _adapter() -> QdrantAdapter:
    return QdrantAdapter(QdrantStoreConfig())


def test_build_filter_none_when_no_payload():
    result = QdrantAdapter._build_filter(None)
    assert result is None


def test_build_filter_none_when_empty_phash_set():
    result = QdrantAdapter._build_filter({"must_not_phash": set()})
    assert result is None


def test_build_filter_returns_qdrant_filter():
    from qdrant_client.models import Filter

    result = QdrantAdapter._build_filter({"must_not_phash": {"abc123", "def456"}})
    assert isinstance(result, Filter)
    assert result.must_not is not None
    assert len(result.must_not) == 1
    condition = result.must_not[0]
    assert condition.key == "image_phash"
    phash_values = set(condition.match.any)
    assert phash_values == {"abc123", "def456"}