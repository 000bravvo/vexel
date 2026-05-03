"""
MultiVectorIndexer — index multiple images per entity into a vector store.

Usage::

    indexer = MultiVectorIndexer(config, encoder, store)
    await indexer.run(
        entities=sku_list,
        entity_id_fn=lambda s: str(s["sku_id"]),
        image_urls_fn=lambda s: [s["img_primary"], s["img_secondary"]],
        extra_payload_fn=lambda s: {"category": s["category"]},
    )

The indexer:
  1. Downloads each image URL (with optional CDN resize).
  2. Optionally detects and skips placeholder images via pHash.
  3. Optionally smart-crops blank space.
  4. Encodes with the provided BaseEncoder.
  5. Upserts to the provided BaseVectorStore in configurable batches.
"""

from __future__ import annotations

import asyncio
import logging
import uuid as _uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from vexel.config import VexelConfig
from vexel.encoder.base import BaseEncoder
from vexel.image_pipeline.fetcher import ImageFetcher
from vexel.image_pipeline.phash_filter import PhashFilter, compute_phash
from vexel.image_pipeline.preprocessor import BlankImageError, Preprocessor
from vexel.indexer.uuid_utils import generate_composite_id
from vexel.store.base import BaseVectorStore, VectorPoint

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class IndexRun:
    """Stats returned from MultiVectorIndexer.run()."""

    total_entities: int = 0
    indexed_vectors: int = 0
    skipped_blank: int = 0
    skipped_placeholder: int = 0
    failed_entities: int = 0
    crop_methods: Counter = field(default_factory=Counter)


class MultiVectorIndexer:
    """
    Parallel, batched indexer for multi-image entities.

    Each entity contributes up to ``config.indexer.max_images_per_entity``
    vectors, one per image type defined in ``config.indexer.image_type_mapping``.

    Thread-safety: the indexer owns its ThreadPoolExecutor.  Create one
    instance per process; it is not safe to share across coroutines running
    concurrently on different event loops.

    Args:
        config:  Root VexelConfig.
        encoder: Pre-instantiated and loaded BaseEncoder.
        store:   Pre-connected BaseVectorStore.
        phash_filter: Optional PhashFilter for placeholder skipping.
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
        self._fetcher = ImageFetcher(
            config.pipeline.cdn_transform, config.pipeline.preprocessor
        )
        self._preprocessor = Preprocessor(config.pipeline.preprocessor)
        self._namespace = _uuid.UUID(config.indexer.namespace_uuid)
        self._executor = ThreadPoolExecutor(
            max_workers=config.indexer.concurrent_workers,
            thread_name_prefix="vexel-indexer",
        )

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    async def run(
        self,
        entities: Iterable[T],
        entity_id_fn: Callable[[T], str],
        image_urls_fn: Callable[[T], list[str]],
        extra_payload_fn: Callable[[T], dict[str, Any]] | None = None,
    ) -> IndexRun:
        """
        Index all entities.

        Args:
            entities:        Iterable of domain objects (dicts, dataclasses, …).
            entity_id_fn:    Extract the string entity ID from an object.
            image_urls_fn:   Extract the ordered list of image URLs.
                             Index 0 → image_type_mapping[0], etc.
            extra_payload_fn: Optional extra fields merged into each point payload.

        Returns:
            IndexRun stats dataclass.
        """
        entity_list = list(entities)
        run = IndexRun(total_entities=len(entity_list))

        buffer: list[VectorPoint] = []
        batch_size = self._cfg.indexer.batch_upsert_size

        for entity in entity_list:
            try:
                points = await self._process_entity(
                    entity, entity_id_fn, image_urls_fn, extra_payload_fn
                )
            except Exception as exc:
                entity_id = entity_id_fn(entity)
                logger.warning("Entity %s failed: %s", entity_id, exc)
                run.failed_entities += 1
                continue

            for point, crop_method, was_skipped_blank, was_skipped_phash in points:
                if was_skipped_blank:
                    run.skipped_blank += 1
                elif was_skipped_phash:
                    run.skipped_placeholder += 1
                elif point is not None:
                    buffer.append(point)
                    run.indexed_vectors += 1
                    if crop_method:
                        run.crop_methods[crop_method] += 1

            if len(buffer) >= batch_size:
                await self._store.upsert(buffer)
                buffer.clear()

        # Flush remainder
        if buffer:
            await self._store.upsert(buffer)

        logger.info(
            "IndexRun complete: %d entities, %d vectors, "
            "%d blank-skipped, %d placeholder-skipped, %d failed",
            run.total_entities,
            run.indexed_vectors,
            run.skipped_blank,
            run.skipped_placeholder,
            run.failed_entities,
        )
        return run

    def close(self) -> None:
        """Shut down the internal thread pool."""
        self._executor.shutdown(wait=False)

    # ---------------------------------------------------------------- #
    # Internal helpers                                                   #
    # ---------------------------------------------------------------- #

    async def _process_entity(
        self,
        entity: T,
        entity_id_fn: Callable[[T], str],
        image_urls_fn: Callable[[T], list[str]],
        extra_payload_fn: Callable[[T], dict[str, Any]] | None,
    ) -> list[tuple[VectorPoint | None, str | None, bool, bool]]:
        """
        Process a single entity and return (point, crop_method, blank, phash_skip)
        tuples for each image slot.
        """
        entity_id = entity_id_fn(entity)
        urls = image_urls_fn(entity)
        extra = extra_payload_fn(entity) if extra_payload_fn else {}
        type_map = self._cfg.indexer.image_type_mapping
        max_imgs = self._cfg.indexer.max_images_per_entity

        tasks = []
        for idx, url in enumerate(urls[:max_imgs]):
            image_type = type_map.get(idx, f"image_{idx}")
            tasks.append(
                self._process_image(entity_id, image_type, url, extra)
            )

        return await asyncio.gather(*tasks)

    async def _process_image(
        self,
        entity_id: str,
        image_type: str,
        url: str,
        extra_payload: dict[str, Any],
    ) -> tuple[VectorPoint | None, str | None, bool, bool]:
        """
        Download, preprocess, encode a single image.

        Returns:
            (VectorPoint | None, crop_method | None, was_blank, was_phash_skip)
        """
        loop = asyncio.get_event_loop()

        try:
            # Download (blocking I/O in thread pool)
            image_bytes = await loop.run_in_executor(
                self._executor,
                self._fetcher.fetch,
                url,
            )
        except Exception as exc:
            logger.debug("Download failed for %s [%s]: %s", entity_id, image_type, exc)
            return None, None, False, False

        if not image_bytes:
            return None, None, False, False

        # pHash placeholder check
        if self._phash_filter:
            phash_hex = await loop.run_in_executor(
                self._executor, compute_phash, image_bytes
            )
            if self._phash_filter.is_placeholder(phash_hex):
                logger.debug(
                    "Placeholder skipped: %s [%s] phash=%s",
                    entity_id, image_type, phash_hex,
                )
                return None, None, False, True
        else:
            phash_hex = None

        # Preprocessing (smart crop) in thread pool
        crop_metadata = None
        crop_method = None
        if self._cfg.pipeline.preprocessor.enabled:
            try:
                import io
                from PIL import Image as PILImage

                pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                pil_img, crop_metadata = await loop.run_in_executor(
                    self._executor, self._preprocessor.process, pil_img
                )
                crop_method = crop_metadata.processing_method if crop_metadata else None
            except BlankImageError:
                logger.debug("Blank image: %s [%s]", entity_id, image_type)
                return None, None, True, False
            except Exception as exc:
                logger.warning(
                    "Preprocess error for %s [%s]: %s", entity_id, image_type, exc
                )
                return None, None, False, False

        # Encode
        try:
            if self._cfg.pipeline.preprocessor.enabled and pil_img is not None:
                vector = await self._encoder.encode_pil(pil_img)
            else:
                vector = await self._encoder.encode(image_bytes)
        except Exception as exc:
            logger.warning("Encode error for %s [%s]: %s", entity_id, image_type, exc)
            return None, None, False, False

        # Build payload
        payload: dict[str, Any] = {
            "entity_id": entity_id,
            "image_type": image_type,
            "original_image_url": url,
        }
        if phash_hex:
            payload["image_phash"] = phash_hex
        if crop_metadata:
            payload["crop_metadata"] = {
                "x": crop_metadata.x,
                "y": crop_metadata.y,
                "w": crop_metadata.w,
                "h": crop_metadata.h,
                "processing_method": crop_metadata.processing_method,
                "was_cropped": crop_metadata.was_cropped,
            }
        payload.update(extra_payload)

        point_id = generate_composite_id(self._namespace, entity_id, image_type)
        point = VectorPoint(id=point_id, vector=vector, payload=payload)

        return point, crop_method, False, False