"""
VisualSearchEngine — query-time image-to-entity search.

Usage::

    engine = VisualSearchEngine(config, encoder, store)
    candidates = await engine.search(image_bytes, top_k=10)
    for c in candidates:
        print(c.entity_id, c.score, c.matched_image_type)

Pipeline:
  1. Encode query image → L2-normalised vector.
  2. ANN retrieval: over-fetch ``top_k × over_fetch_multiplier`` raw hits
     (catches multi-vector duplicates and placeholder noise).
  3. Optional pHash filter on the ANN filter_payload (pre-filter in store).
  4. Deduplicate by entity_id, keeping max score and matched_image_type.
  5. Apply min_score filter.
  6. Apply score-gap thresholding.
  7. Return top ``max_results`` SearchCandidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vexel.config import VexelConfig
from vexel.encoder.base import BaseEncoder
from vexel.image_pipeline.phash_filter import PhashFilter, compute_phash
from vexel.search.threshold import apply_score_gap_threshold
from vexel.store.base import BaseVectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchCandidate:
    """A single deduplicated result from VisualSearchEngine.search()."""

    entity_id: str
    """Domain entity identifier (SKU ID, product ID, …)."""

    score: float
    """Maximum cosine similarity across all matched image vectors."""

    matched_image_type: str
    """Image type (e.g. "primary") of the highest-scoring hit."""

    payload: dict[str, Any] = field(default_factory=dict)
    """Full payload from the winning vector point (includes crop_metadata, etc.)."""


class VisualSearchEngine:
    """
    Domain-agnostic visual search engine.

    Args:
        config:       Root VexelConfig.
        encoder:      Loaded BaseEncoder.
        store:        Connected BaseVectorStore.
        phash_filter: Optional PhashFilter.  When provided, the query image's
                      pHash is used to pre-filter placeholder vectors from ANN
                      retrieval (avoids returning results that match a placeholder
                      rather than a real product image).
    """

    def __init__(
        self,
        config: VexelConfig,
        encoder: BaseEncoder,
        store: BaseVectorStore,
        phash_filter: PhashFilter | None = None,
    ) -> None:
        self._cfg = config
        self._encoder = encoder
        self._store = store
        self._phash_filter = phash_filter

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    async def search(
        self,
        image_bytes: bytes,
        top_k: int = 10,
    ) -> list[SearchCandidate]:
        """
        Find the most visually similar entities to the query image.

        Args:
            image_bytes: Raw bytes of the query image (JPEG, PNG, WebP, …).
            top_k:       Number of deduplicated results to target.

        Returns:
            Ranked list of SearchCandidates (highest score first).
            May contain fewer than ``top_k`` items if not enough results
            pass the score / gap filters.
        """
        search_cfg = self._cfg.search

        # 1. Build pHash filter payload (optional)
        filter_payload = await self._build_filter_payload(image_bytes)

        # 2. Encode query
        query_vector = await self._encoder.encode(image_bytes)

        # 3. Over-fetch ANN hits
        raw_limit = top_k * search_cfg.over_fetch_multiplier
        raw_hits = await self._store.search(
            query_vector=query_vector,
            limit=raw_limit,
            filter_payload=filter_payload,
        )

        logger.debug("ANN returned %d raw hits for top_k=%d", len(raw_hits), top_k)

        # 4. Deduplicate by entity_id (keep max score)
        best: dict[str, tuple[float, str, dict[str, Any]]] = {}
        for hit in raw_hits:
            eid = hit.payload.get("entity_id", hit.id)
            img_type = hit.payload.get("image_type", "")
            prev_score, *_ = best.get(eid, (float("-inf"), "", {}))
            if hit.score > prev_score:
                best[eid] = (hit.score, img_type, hit.payload)

        # 5. Sort descending
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)

        # 6. Min score filter
        ranked = [
            (eid, (s, t, p))
            for eid, (s, t, p) in ranked
            if s >= search_cfg.min_score
        ]

        # 7. Score-gap threshold
        scores = [s for _, (s, _, _) in ranked]
        keep = apply_score_gap_threshold(scores, search_cfg.score_gap_threshold)
        ranked = ranked[:keep]

        # 8. Hard cap
        ranked = ranked[: search_cfg.max_results]

        return [
            SearchCandidate(
                entity_id=eid,
                score=round(score, 6),
                matched_image_type=img_type,
                payload=payload,
            )
            for eid, (score, img_type, payload) in ranked
        ]

    # ---------------------------------------------------------------- #
    # Internal helpers                                                   #
    # ---------------------------------------------------------------- #

    async def _build_filter_payload(
        self, image_bytes: bytes
    ) -> dict[str, Any] | None:
        """
        If a PhashFilter is attached, compute the query image's pHash and
        return a filter that excludes vectors sharing that pHash (i.e., the
        query image itself is a placeholder).

        In practice this also avoids returning other placeholder images when
        the query happens to be one.
        """
        if not self._phash_filter:
            return None

        try:
            phash_hex = compute_phash(image_bytes)
            if self._phash_filter.is_placeholder(phash_hex):
                logger.debug("Query image is a placeholder (phash=%s)", phash_hex)
                # Exclude all known placeholder hashes from ANN results
                known_hashes = self._phash_filter.get_all_hashes()
                return {"must_not_phash": known_hashes}
        except Exception as exc:
            logger.warning("pHash computation failed: %s", exc)

        return None