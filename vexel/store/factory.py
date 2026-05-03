"""
Store factory — resolves a backend string to a connected BaseVectorStore.
"""

from __future__ import annotations

from vexel.config import StoreConfig
from vexel.store.base import BaseVectorStore


def get_store(config: StoreConfig) -> BaseVectorStore:
    """
    Instantiate the vector store adapter specified in *config.backend*.

    Supported backends:
        "qdrant"    — QdrantAdapter (requires vexel[qdrant])
        "pinecone"  — PineconeAdapter stub (requires vexel[pinecone])
        "weaviate"  — WeaviateAdapter stub (requires weaviate-client>=4.0)

    The returned adapter is **not** yet connected; call ``await store.connect()``
    before use.
    """
    backend = config.backend.lower()

    if backend == "qdrant":
        from vexel.store.qdrant import QdrantAdapter

        if config.qdrant is None:
            raise ValueError(
                "store.backend is 'qdrant' but store.qdrant config block is missing."
            )
        return QdrantAdapter(config.qdrant)

    if backend == "pinecone":
        from vexel.store.pinecone import PineconeAdapter

        if config.pinecone is None:
            raise ValueError(
                "store.backend is 'pinecone' but store.pinecone config block is missing."
            )
        return PineconeAdapter(
            api_key=config.pinecone.api_key,
            index_name=config.pinecone.index_name,
            namespace=config.pinecone.namespace,
        )

    if backend == "weaviate":
        from vexel.store.weaviate import WeaviateAdapter

        if config.weaviate is None:
            raise ValueError(
                "store.backend is 'weaviate' but store.weaviate config block is missing."
            )
        return WeaviateAdapter(
            url=config.weaviate.url,
            class_name=config.weaviate.class_name,
            api_key=config.weaviate.api_key,
        )

    raise ValueError(
        f"Unknown store backend '{config.backend}'.  "
        "Supported: 'qdrant', 'pinecone', 'weaviate'."
    )