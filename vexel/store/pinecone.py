"""
PineconeAdapter — BaseVectorStore stub for Pinecone.

Requires: pip install vexel[pinecone]

This is a stub implementation.  Wire up the Pinecone gRPC/HTTP client once
pinecone-client v3+ is added to the optional deps.
"""

from __future__ import annotations

import logging
from typing import Any

from vexel.store.base import BaseVectorStore, RawHit, VectorPoint

logger = logging.getLogger(__name__)


class PineconeAdapter(BaseVectorStore):
    """
    Pinecone adapter (stub — not yet implemented).

    Raises NotImplementedError for all operations until a full implementation
    is contributed.  The interface follows BaseVectorStore exactly so that
    drop-in replacement requires only config + adapter swap.
    """

    def __init__(self, api_key: str, index_name: str, namespace: str = "") -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace

    async def connect(self) -> None:
        raise NotImplementedError(
            "PineconeAdapter is not yet implemented.  "
            "Contributions welcome: https://github.com/vexel/vexel"
        )

    async def close(self) -> None:
        pass  # nothing to close in stub

    async def upsert(self, points: list[VectorPoint]) -> None:
        raise NotImplementedError("PineconeAdapter.upsert() not implemented")

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[RawHit]:
        raise NotImplementedError("PineconeAdapter.search() not implemented")