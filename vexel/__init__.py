"""
vexel — domain-agnostic multi-vector image search library.

Quick start::

    from vexel import VexelConfig, MultiVectorIndexer, VisualSearchEngine, SearchCandidate
    from vexel.encoder.factory import get_encoder
    from vexel.store.factory import get_store

    config  = VexelConfig.from_yaml("vexel.yaml")
    encoder = get_encoder(config.encoder)
    store   = get_store(config.store)

    await encoder.load()
    await store.connect()

    # Index
    indexer = MultiVectorIndexer(config, encoder, store)
    await indexer.run(entities=..., entity_id_fn=..., image_urls_fn=...)

    # Search
    engine  = VisualSearchEngine(config, encoder, store)
    results = await engine.search(image_bytes, top_k=10)
    # results[i].entity_id, results[i].score, results[i].matched_image_type
"""

from vexel.config import (
    VexelConfig,
    EncoderConfig,
    IndexerConfig,
    PipelineConfig,
    PineconeStoreConfig,
    PlaceholderFilterConfig,
    QdrantStoreConfig,
    SearchConfig,
    StoreConfig,
    WeaviateStoreConfig,
)
from vexel.indexer.multi_vector import MultiVectorIndexer
from vexel.search.engine import SearchCandidate, VisualSearchEngine

__all__ = [
    # Config
    "VexelConfig",
    "EncoderConfig",
    "StoreConfig",
    "QdrantStoreConfig",
    "PineconeStoreConfig",
    "WeaviateStoreConfig",
    "PipelineConfig",
    "IndexerConfig",
    "SearchConfig",
    "PlaceholderFilterConfig",
    # Core API
    "MultiVectorIndexer",
    "VisualSearchEngine",
    "SearchCandidate",
]

__version__ = "0.1.0"