"""
VexelConfig — Pydantic-based configuration with YAML loader.

Integration contract: write a vexel.yaml, call VexelConfig.from_yaml().
No Python config code required.

Example:

    config = VexelConfig.from_yaml("vexel.yaml")
    # or programmatically:
    config = VexelConfig(
        encoder=EncoderConfig(model="ViT-B/32"),
        store=StoreConfig(backend="qdrant"),
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typing import Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Sub-configs                                                          #
# ------------------------------------------------------------------ #


class EncoderConfig(BaseModel):
    """Which embedding model to use."""

    name: str = "clip"
    """Encoder name: "clip" | "siglip" | "custom"."""

    model: str = "ViT-B/32"
    """Model variant, passed to the encoder implementation."""

    device: str = "cpu"
    """Inference device: "cpu" | "cuda" | "mps"."""

    batch_size: int = 8
    """Images per encode_batch() call."""


class QdrantStoreConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    collection: str = "entity_images"
    vector_size: int = 512


class PineconeStoreConfig(BaseModel):
    api_key: str = ""
    index_name: str = "entity-images"
    namespace: str = ""
    """Pinecone index namespace (empty = default)."""


class WeaviateStoreConfig(BaseModel):
    url: str = "http://localhost:8080"
    class_name: str = "EntityImage"
    api_key: str = ""


class StoreConfig(BaseModel):
    """Vector store backend selection."""

    backend: str = "qdrant"
    """Backend name: "qdrant" | "pinecone" | "weaviate"."""

    qdrant: QdrantStoreConfig = Field(default_factory=QdrantStoreConfig)
    pinecone: Optional[PineconeStoreConfig] = None
    weaviate: Optional[WeaviateStoreConfig] = None


class PreprocessorConfig(BaseModel):
    """Controls the smart-crop blank-space removal pipeline."""

    enabled: bool = True
    """When False, images are passed to the encoder without cropping."""

    download_size: int = 512
    """Resolution to download via CDN before cropping (more pixels = better
    contour detection). Set to 0 to disable CDN resize."""

    target_size: int = 224
    """Final square side length passed to the encoder."""

    padding_color: list[int] = Field(default_factory=lambda: [128, 128, 128])
    """RGB fill colour for the square padding canvas (neutral grey)."""

    bg_threshold: float = 15.0
    """Euclidean RGB distance threshold for solid-colour BG detection (Tier 1)."""

    near_neutral_tolerance: int = 10
    """Pixel tolerance for the near-white / near-black fast-track (Tier 2)."""

    margin_pct: float = 0.05
    """Expand the detected bounding box by this fraction on each side."""

    min_subject_area: float = 0.05
    """Minimum subject area fraction; below this, skip crop and use full image."""


class CDNTransformConfig(BaseModel):
    """Controls CDN URL rewriting for resize-on-the-fly."""

    enabled: bool = True
    """When False, raw image URLs are used as-is."""

    template: str = "{url}?w={size}&h={size}"
    """URL template.  Placeholders: {url}, {size}.
    Gumlet:     "{url}?w={size}&h={size}"
    Cloudinary: "{url}/c_fit,w_{size},h_{size}"
    Imgix:      "{url}?w={size}&h={size}&fit=clip"
    """


class PipelineConfig(BaseModel):
    """Image pipeline settings (download + preprocess)."""

    preprocessor: PreprocessorConfig = Field(default_factory=PreprocessorConfig)
    cdn_transform: CDNTransformConfig = Field(default_factory=CDNTransformConfig)


class PlaceholderFilterConfig(BaseModel):
    """pHash-based placeholder image detection & filtering."""

    registry_path: str = "./placeholder_hashes.json"
    """Path to the JSON registry of known placeholder pHashes."""

    frequency_threshold: int = 3
    """Images whose pHash appears on ≥ this many entities are auto-promoted
    to the registry during post-run analysis."""


class IndexerConfig(BaseModel):
    """MultiVectorIndexer behaviour."""

    max_images_per_entity: int = 3
    """Hard cap on images indexed per entity."""

    concurrent_workers: int = 8
    """ThreadPoolExecutor size for parallel download + encode."""

    batch_upsert_size: int = 100
    """Number of VectorPoints accumulated before a Qdrant upsert call."""

    namespace_uuid: str = "12345678-1234-5678-1234-567812345678"
    """UUID5 namespace.  MUST NOT change after the first index run."""

    image_type_mapping: dict[int, str] = Field(
        default_factory=lambda: {0: "primary", 1: "secondary", 2: "tertiary"}
    )
    """Maps image array position → semantic label stored in payload.
    Override per domain, e.g. {0: "primary_box", 1: "blister_strip", 2: "loose_pill"}
    or {0: "main_shot", 1: "side_view", 2: "sole_detail"}.
    """


class SearchConfig(BaseModel):
    """VisualSearchEngine behaviour."""

    over_fetch_multiplier: int = 3
    """Raw Qdrant hits = top_k * this value (absorbs multi-vector duplicates)."""

    score_gap_threshold: float = 0.08
    """Drop results after the first score gap larger than this value."""

    min_score: float = 0.70
    """Absolute minimum cosine similarity to include in results."""

    max_results: int = 20
    """Hard cap on results returned regardless of top_k."""


# ------------------------------------------------------------------ #
# Root config                                                          #
# ------------------------------------------------------------------ #


class VexelConfig(BaseModel):
    """
    Root configuration object.

    Load from YAML:
        config = VexelConfig.from_yaml("vexel.yaml")

    The YAML file must have a top-level "vexel:" key, or be the config
    dict itself (backwards-compatible).
    """

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    placeholder_filter: PlaceholderFilterConfig = Field(
        default_factory=PlaceholderFilterConfig
    )
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VexelConfig":
        """
        Load config from a YAML file.

        The file may contain either:
          - a top-level "vexel:" key whose value is the config dict
          - the config dict directly at the top level

        Args:
            path: path to the YAML config file.

        Returns:
            Validated VexelConfig instance.
        """
        import yaml  # deferred to keep startup import light

        data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        # Support both "vexel:" nested and flat config files
        vexel_data = data.get("vexel", data)
        return cls.model_validate(vexel_data)