"""
WeaviateAdapter — BaseVectorStore stub for Weaviate.

Requires: pip install weaviate-client>=4.0

This is a stub implementation.  Wire up the Weaviate async client once
weaviate-client v4+ is added to the optional deps.
"""

from __future__ import annotations

import logging
from typing import Any

from vexel.store.base import BaseVectorStore, RawHit, VectorPoint

logger = logging.getLogger(__name__)


class WeaviateAdapter(BaseVectorStore):
    """
    Weaviate adapter (stub — not yet implemented).

    Raises NotImplementedError for all operations until a full implementation
    is contributed.
    """

    def __init__(self, url: str, class_name: str, api_key: str = "") -> None:
        self._url = url
        self._class_name = class_name
        self._api_key = api_key

    async def connect(self) -> None:
        raise NotImplementedError(
            "WeaviateAdapter is not yet implemented.  "
            "Contributions welcome: https://github.com/vexel/vexel"
        )

    async def close(self) -> None:
        pass

    async def upsert(self, points: list[VectorPoint]) -> None:
        raise NotImplementedError("WeaviateAdapter.upsert() not implemented")

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[RawHit]:
        raise NotImplementedError("WeaviateAdapter.search() not implemented")