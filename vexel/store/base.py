"""
BaseVectorStore — abstract interface for vector database backends.

Implement this to add Pinecone, Weaviate, Milvus, or any other vector store
without touching any indexer or search code.

VectorPoint and RawHit are the data contracts crossing the boundary between
vexel's domain logic and the storage layer.  Both are plain dataclasses with
no backend-specific types, so they're easy to test without a running database.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any


@dataclasses.dataclass
class VectorPoint:
    """
    A single vector to be stored.

    id      — deterministic UUID5 string (see vexel.indexer.uuid_utils)
    vector  — L2-normalised float32 embedding
    payload — arbitrary metadata dict stored alongside the vector;
              must include at minimum {"entity_id": str, "image_type": str}
    """

    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclasses.dataclass
class RawHit:
    """
    A single ANN search result from the vector store, before deduplication.

    id      — point ID
    score   — cosine similarity (0.0–1.0)
    payload — metadata stored with the vector (e.g. entity_id, image_type)
    """

    id: str
    score: float
    payload: dict[str, Any]


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store backends.

    All methods are async.  Implementations must ensure that connect()
    is idempotent (safe to call more than once) and that close() is
    safe to call even if connect() was never called.

    Example::

        class MyStore(BaseVectorStore):
            async def connect(self) -> None: ...
            async def upsert(self, points): ...
            async def search(self, vector, limit, filter_payload=None): ...
            async def close(self) -> None: ...
    """

    # ---------------------------------------------------------------- #
    # Lifecycle                                                          #
    # ---------------------------------------------------------------- #

    @abstractmethod
    async def connect(self) -> None:
        """
        Initialise the client connection and verify reachability.

        Should create the collection / index if it doesn't exist.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release all connections and resources."""

    # ---------------------------------------------------------------- #
    # Write                                                              #
    # ---------------------------------------------------------------- #

    @abstractmethod
    async def upsert(self, points: list[VectorPoint]) -> None:
        """
        Insert or update a batch of vectors.

        Implementations should be idempotent: upserting the same point ID
        twice must update the existing record, not create a duplicate.

        Args:
            points: list of VectorPoint to upsert.
        """

    # ---------------------------------------------------------------- #
    # Read                                                               #
    # ---------------------------------------------------------------- #

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[RawHit]:
        """
        Run approximate nearest-neighbour search.

        Args:
            query_vector:   L2-normalised query embedding.
            limit:          maximum number of raw hits to return.
                            The caller (VisualSearchEngine) over-fetches
                            and deduplicates, so this is larger than top_k.
            filter_payload: optional backend-specific filter dict.
                            The engine passes {"must_not_phash": set[str]}
                            when placeholder filtering is needed.
                            Adapters translate this to their native format.

        Returns:
            list[RawHit] ordered by score descending.
        """

    # ---------------------------------------------------------------- #
    # Optional: collection management                                   #
    # ---------------------------------------------------------------- #

    async def scroll(
        self,
        limit: int = 1000,
        offset: Any | None = None,
        with_payload: list[str] | None = None,
    ) -> tuple[list[RawHit], Any | None]:
        """
        Page through all stored points (for incremental indexing).

        Returns (hits, next_offset).  next_offset is None when exhausted.
        Default raises NotImplementedError; override in adapters that
        support incremental indexing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement scroll(). "
            "Override this method to support incremental indexing."
        )