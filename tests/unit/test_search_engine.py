"""Unit tests for VisualSearchEngine result deduplication and filtering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vexel.config import VexelConfig
from vexel.search.engine import SearchCandidate, VisualSearchEngine
from vexel.store.base import RawHit


def _make_engine(config: VexelConfig | None = None) -> tuple[VisualSearchEngine, MagicMock, MagicMock]:
    from vexel.config import SearchConfig
    # Use permissive defaults so individual tests can focus on one behaviour
    cfg = config or VexelConfig(
        search=SearchConfig(score_gap_threshold=1.0, min_score=0.0)
    )
    encoder = MagicMock()
    encoder.encode = AsyncMock(return_value=[0.1] * 512)
    store = MagicMock()
    engine = VisualSearchEngine(cfg, encoder, store)
    return engine, encoder, store


@pytest.mark.asyncio
async def test_deduplication_keeps_max_score():
    # Use a config with no gap/score filtering so dedup is the only logic tested
    from vexel.config import SearchConfig
    engine, encoder, store = _make_engine(
        VexelConfig(search=SearchConfig(score_gap_threshold=1.0, min_score=0.0))
    )
    store.search = AsyncMock(
        return_value=[
            RawHit(id="uuid-1", score=0.90, payload={"entity_id": "SKU-1", "image_type": "primary"}),
            RawHit(id="uuid-2", score=0.85, payload={"entity_id": "SKU-1", "image_type": "secondary"}),
            RawHit(id="uuid-3", score=0.80, payload={"entity_id": "SKU-2", "image_type": "primary"}),
        ]
    )

    results = await engine.search(b"fake-image", top_k=10)

    assert len(results) == 2
    sku1 = next(r for r in results if r.entity_id == "SKU-1")
    assert sku1.score == pytest.approx(0.90)
    assert sku1.matched_image_type == "primary"


@pytest.mark.asyncio
async def test_min_score_filter():
    from vexel.config import SearchConfig

    cfg = VexelConfig(search=SearchConfig(min_score=0.85, score_gap_threshold=0.0))
    engine, encoder, store = _make_engine(cfg)
    store.search = AsyncMock(
        return_value=[
            RawHit(id="1", score=0.95, payload={"entity_id": "A", "image_type": "primary"}),
            RawHit(id="2", score=0.80, payload={"entity_id": "B", "image_type": "primary"}),
        ]
    )

    results = await engine.search(b"img", top_k=5)
    assert len(results) == 1
    assert results[0].entity_id == "A"


@pytest.mark.asyncio
async def test_score_gap_threshold_applied():
    from vexel.config import SearchConfig

    cfg = VexelConfig(search=SearchConfig(min_score=0.0, score_gap_threshold=0.08))
    engine, encoder, store = _make_engine(cfg)
    store.search = AsyncMock(
        return_value=[
            RawHit(id="1", score=0.95, payload={"entity_id": "A", "image_type": "primary"}),
            RawHit(id="2", score=0.93, payload={"entity_id": "B", "image_type": "primary"}),
            RawHit(id="3", score=0.75, payload={"entity_id": "C", "image_type": "primary"}),
        ]
    )

    results = await engine.search(b"img", top_k=10)
    # Gap: 0.93→0.75 = 0.18 > 0.08 → only A and B kept
    assert len(results) == 2
    assert {r.entity_id for r in results} == {"A", "B"}


@pytest.mark.asyncio
async def test_empty_results():
    engine, encoder, store = _make_engine()
    store.search = AsyncMock(return_value=[])

    results = await engine.search(b"img", top_k=10)
    assert results == []


@pytest.mark.asyncio
async def test_results_sorted_descending():
    engine, encoder, store = _make_engine()
    store.search = AsyncMock(
        return_value=[
            RawHit(id="3", score=0.80, payload={"entity_id": "C", "image_type": "primary"}),
            RawHit(id="1", score=0.95, payload={"entity_id": "A", "image_type": "primary"}),
            RawHit(id="2", score=0.88, payload={"entity_id": "B", "image_type": "primary"}),
        ]
    )

    results = await engine.search(b"img", top_k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)