"""
UUID utilities for stable, reproducible vector point IDs.

Design principle:
    Each (entity_id, image_type) pair maps to a deterministic UUID5 so that
    re-indexing an entity always overwrites the previous vector (upsert is
    idempotent).  The namespace UUID must never change after the first run.
"""

from __future__ import annotations

import uuid as _uuid


def generate_composite_id(
    namespace: str | _uuid.UUID,
    entity_id: str,
    image_type: str,
) -> str:
    """
    Generate a stable UUID5 string for a (entity_id, image_type) pair.

    Args:
        namespace: UUID5 namespace string or UUID object.  This MUST remain
                   constant across all index runs for idempotent upserts.
        entity_id: Domain entity identifier (e.g. "SKU-12345").
        image_type: Semantic label for this image slot (e.g. "primary").

    Returns:
        Lowercase hyphenated UUID string, e.g.
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8".

    Example::

        point_id = generate_composite_id(
            "12345678-1234-5678-1234-567812345678",
            "SKU-42",
            "primary",
        )
    """
    if isinstance(namespace, str):
        namespace = _uuid.UUID(namespace)
    name = f"{entity_id}::{image_type}"
    return str(_uuid.uuid5(namespace, name))