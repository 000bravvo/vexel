"""
PhashFilter — perceptual hash registry for placeholder image detection.

Problem: many catalog images share a stock "no image available" graphic.
Indexing them produces vectors that attract irrelevant results.

Solution:
  1. Compute pHash for every indexed image.
  2. Post-run: images whose hash appears on ≥ N distinct entities are
     "placeholders" and are promoted to the registry JSON.
  3. At search time: filter out Qdrant hits whose image_phash is in
     the in-memory frozenset (zero query overhead).

The registry JSON format:
  {
    "<phash_hex>": {
      "entity_count": 12,
      "promoted_at": "2026-05-03"
    },
    ...
  }
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from vexel.config import PlaceholderFilterConfig

logger = logging.getLogger(__name__)


def compute_phash(image_bytes: bytes) -> str:
    """
    Return the 64-bit perceptual hash of an image as a hex string.

    Uses imagehash.phash — robust to minor scale / quality changes.
    """
    import imagehash
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return str(imagehash.phash(img))


class PhashFilter:
    """
    Manages the placeholder pHash registry.

    Lifecycle:
        filter = PhashFilter(config.placeholder_filter)
        filter.load()                        # at startup / indexer init

        # during indexing:
        if filter.is_placeholder(phash):
            continue  # skip this image

        # after indexing:
        filter.update_registry(phash_list, entity_ids)  # auto-promote
    """

    def __init__(self, config: PlaceholderFilterConfig) -> None:
        self._config = config
        self._registry_path = Path(config.registry_path)
        self._hashes: frozenset[str] = frozenset()

    # ---------------------------------------------------------------- #
    # Load / persist                                                     #
    # ---------------------------------------------------------------- #

    def load(self) -> None:
        """
        Load the placeholder hashes from the JSON registry into memory.

        Missing or empty registry is treated as "no placeholders known" —
        not an error.
        """
        if not self._registry_path.exists():
            logger.info(
                "Placeholder registry not found at %s — skipping",
                self._registry_path,
            )
            self._hashes = frozenset()
            return

        data: dict = json.loads(self._registry_path.read_text())
        self._hashes = frozenset(data.keys())
        logger.info(
            "Loaded %d placeholder hashes from %s",
            len(self._hashes),
            self._registry_path,
        )

    def get_all_hashes(self) -> frozenset[str]:
        """Return the currently loaded frozenset of placeholder pHashes."""
        return self._hashes

    # ---------------------------------------------------------------- #
    # Query                                                              #
    # ---------------------------------------------------------------- #

    def is_placeholder(self, phash: str) -> bool:
        """Return True if this pHash is a known placeholder."""
        return phash in self._hashes

    # ---------------------------------------------------------------- #
    # Registry maintenance                                               #
    # ---------------------------------------------------------------- #

    def update_registry(
        self,
        phash_entity_pairs: list[tuple[str, str]],
    ) -> dict[str, int]:
        """
        Scan pHash frequencies and promote frequent hashes to the registry.

        Args:
            phash_entity_pairs: list of (phash_hex, entity_id) tuples from
                                the current index run.  One tuple per indexed
                                image point.

        Returns:
            dict {"added": int, "updated": int, "total": int}
        """
        import datetime as dt

        # Count how many *distinct entities* each phash appears on
        phash_to_entities: dict[str, set[str]] = {}
        for ph, eid in phash_entity_pairs:
            if ph:
                phash_to_entities.setdefault(ph, set()).add(eid)

        entity_counts: Counter = Counter(
            {ph: len(eids) for ph, eids in phash_to_entities.items()}
        )

        existing: dict = {}
        if self._registry_path.exists():
            existing = json.loads(self._registry_path.read_text())

        added = updated = 0
        threshold = self._config.frequency_threshold

        for ph, count in entity_counts.items():
            if count < threshold:
                continue
            if ph not in existing:
                existing[ph] = {
                    "entity_count": count,
                    "promoted_at": dt.date.today().isoformat(),
                }
                added += 1
            elif existing[ph].get("entity_count", 0) != count:
                existing[ph]["entity_count"] = count
                updated += 1

        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(json.dumps(existing, indent=2))

        # Refresh in-memory frozenset
        self._hashes = frozenset(existing.keys())

        stats = {"added": added, "updated": updated, "total": len(existing)}
        logger.info("Placeholder registry updated: %s", stats)
        return stats