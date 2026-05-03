"""
QdrantAdapter — BaseVectorStore implementation for Qdrant.

Requires: pip install vexel[qdrant]

Handles:
  - Collection auto-creation on connect()
  - Batch upsert via qdrant_client PointStruct
  - ANN search with optional placeholder pHash filter (must_not payload filter)
  - scroll() for incremental indexing
"""

from __future__ import annotations

import logging
from typing import Any

from vexel.config import QdrantStoreConfig
from vexel.store.base import BaseVectorStore, RawHit, VectorPoint

logger = logging.getLogger(__name__)


class QdrantAdapter(BaseVectorStore):
    """
    Async Qdrant adapter.

    Wraps qdrant_client.AsyncQdrantClient.  The underlying client handles
    connection pooling; connect() performs a startup health-check so a
    misconfigured Qdrant raises at boot, not at query time.

    Example::

        store = QdrantAdapter(config.store.qdrant)
        await store.connect()
        await store.upsert(points)
        hits = await store.search(query_vector, limit=60)
        await store.close()
    """

    def __init__(self, config: QdrantStoreConfig) -> None:
        self._config = config
        self._client = None

    # ---------------------------------------------------------------- #
    # Lifecycle                                                          #
    # ---------------------------------------------------------------- #

    async def connect(self) -> None:
        """
        Connect to Qdrant and ensure the collection exists.

        Creates the collection with cosine HNSW if it does not already exist.
        """
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        cfg = self._config
        logger.info("Connecting to Qdrant at %s:%s", cfg.host, cfg.port)
        self._client = AsyncQdrantClient(
            host=cfg.host,
            port=cfg.port,
            grpc_port=cfg.grpc_port,
            prefer_grpc=cfg.prefer_grpc,
        )

        # Health check — raises ConnectionError if unreachable
        await self._client.get_collections()
        logger.info("Qdrant connection established")

        # Ensure collection exists (idempotent)
        existing = {c.name for c in (await self._client.get_collections()).collections}
        if cfg.collection not in existing:
            await self._client.create_collection(
                collection_name=cfg.collection,
                vectors_config=VectorParams(
                    size=cfg.vector_size, distance=Distance.COSINE
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                cfg.collection,
                cfg.vector_size,
            )
        else:
            logger.info("Qdrant collection '%s' already exists", cfg.collection)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Qdrant connection closed")

    # ---------------------------------------------------------------- #
    # Write                                                              #
    # ---------------------------------------------------------------- #

    async def upsert(self, points: list[VectorPoint]) -> None:
        """Batch-upsert VectorPoints into the configured collection."""
        from qdrant_client.models import PointStruct

        self._assert_connected()
        structs = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        await self._client.upsert(
            collection_name=self._config.collection, points=structs
        )

    # ---------------------------------------------------------------- #
    # Search                                                             #
    # ---------------------------------------------------------------- #

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[RawHit]:
        """
        Run ANN search against the collection.

        filter_payload supports:
            {"must_not_phash": set[str] | frozenset[str]}
            Translates to a Qdrant must_not FieldCondition on "image_phash".

        Returns list[RawHit] ordered by score descending.
        """
        self._assert_connected()
        query_filter = self._build_filter(filter_payload)

        results = await self._client.search(
            collection_name=self._config.collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )

        return [
            RawHit(
                id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {},
            )
            for hit in results
        ]

    # ---------------------------------------------------------------- #
    # Scroll (incremental indexing)                                      #
    # ---------------------------------------------------------------- #

    async def scroll(
        self,
        limit: int = 1000,
        offset: Any | None = None,
        with_payload: list[str] | None = None,
    ) -> tuple[list[RawHit], Any | None]:
        """
        Page through stored points for incremental index runs.

        Returns:
            (hits, next_offset) — next_offset is None when exhausted.
        """
        self._assert_connected()
        result, next_offset = await self._client.scroll(
            collection_name=self._config.collection,
            scroll_filter=None,
            limit=limit,
            offset=offset,
            with_payload=with_payload or True,
            with_vectors=False,
        )
        hits = [
            RawHit(id=str(p.id), score=0.0, payload=p.payload or {})
            for p in result
        ]
        return hits, next_offset

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    @property
    def is_ready(self) -> bool:
        """True once connect() has succeeded."""
        return self._client is not None

    def _assert_connected(self) -> None:
        if self._client is None:
            raise RuntimeError(
                "QdrantAdapter: connect() must be called before use."
            )

    @staticmethod
    def _build_filter(filter_payload: dict[str, Any] | None):
        """Translate the generic filter_payload dict to a Qdrant Filter."""
        if not filter_payload:
            return None

        must_not_phash = filter_payload.get("must_not_phash")
        if not must_not_phash:
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchAny

        return Filter(
            must_not=[
                FieldCondition(
                    key="image_phash",
                    match=MatchAny(any=list(must_not_phash)),
                )
            ]
        )