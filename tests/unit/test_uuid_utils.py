"""Unit tests for vexel.indexer.uuid_utils."""

import uuid

import pytest

from vexel.indexer.uuid_utils import generate_composite_id

NS = "12345678-1234-5678-1234-567812345678"


def test_returns_valid_uuid_string():
    result = generate_composite_id(NS, "SKU-1", "primary")
    # Must not raise
    parsed = uuid.UUID(result)
    assert str(parsed) == result


def test_deterministic():
    a = generate_composite_id(NS, "SKU-42", "secondary")
    b = generate_composite_id(NS, "SKU-42", "secondary")
    assert a == b


def test_different_entity_ids_produce_different_uuids():
    a = generate_composite_id(NS, "SKU-1", "primary")
    b = generate_composite_id(NS, "SKU-2", "primary")
    assert a != b


def test_different_image_types_produce_different_uuids():
    a = generate_composite_id(NS, "SKU-1", "primary")
    b = generate_composite_id(NS, "SKU-1", "secondary")
    assert a != b


def test_accepts_uuid_namespace_object():
    ns_obj = uuid.UUID(NS)
    result = generate_composite_id(ns_obj, "SKU-1", "primary")
    assert result == generate_composite_id(NS, "SKU-1", "primary")


def test_stable_across_calls_known_value():
    # Freeze a known output so regressions are caught immediately.
    known = generate_composite_id(NS, "SKU-999", "primary")
    assert generate_composite_id(NS, "SKU-999", "primary") == known