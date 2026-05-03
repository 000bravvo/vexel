"""vexel.store — vector store adapters."""

from vexel.store.base import BaseVectorStore, VectorPoint, RawHit
from vexel.store.factory import get_store

__all__ = ["BaseVectorStore", "VectorPoint", "RawHit", "get_store"]