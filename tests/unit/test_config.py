"""Unit tests for VexelConfig YAML loading and defaults."""

import textwrap
from pathlib import Path

import pytest

from vexel.config import VexelConfig


def _write_yaml(tmp_path: Path, content: str) -> Path:
    f = tmp_path / "vexel.yaml"
    f.write_text(textwrap.dedent(content))
    return f


def test_defaults_when_no_yaml(tmp_path):
    f = _write_yaml(tmp_path, "{}")
    cfg = VexelConfig.from_yaml(f)
    assert cfg.encoder.name == "clip"
    assert cfg.store.backend == "qdrant"
    assert cfg.pipeline.preprocessor.enabled is True


def test_nested_vexel_key(tmp_path):
    f = _write_yaml(
        tmp_path,
        """
        vexel:
          encoder:
            name: clip
            model: ViT-L/14
            device: cuda
          store:
            backend: qdrant
            qdrant:
              collection: my_collection
        """,
    )
    cfg = VexelConfig.from_yaml(f)
    assert cfg.encoder.model == "ViT-L/14"
    assert cfg.encoder.device == "cuda"
    assert cfg.store.qdrant.collection == "my_collection"


def test_flat_yaml(tmp_path):
    f = _write_yaml(
        tmp_path,
        """
        encoder:
          name: clip
          model: ViT-B/32
        store:
          backend: qdrant
        """,
    )
    cfg = VexelConfig.from_yaml(f)
    assert cfg.encoder.model == "ViT-B/32"


def test_image_type_mapping_override(tmp_path):
    f = _write_yaml(
        tmp_path,
        """
        vexel:
          indexer:
            image_type_mapping:
              0: main_shot
              1: side_view
              2: sole_detail
        """,
    )
    cfg = VexelConfig.from_yaml(f)
    assert cfg.indexer.image_type_mapping[0] == "main_shot"
    assert cfg.indexer.image_type_mapping[2] == "sole_detail"


def test_search_defaults(tmp_path):
    f = _write_yaml(tmp_path, "{}")
    cfg = VexelConfig.from_yaml(f)
    assert cfg.search.score_gap_threshold == 0.08
    assert cfg.search.min_score == 0.70
    assert cfg.search.over_fetch_multiplier == 3


def test_store_pinecone_config(tmp_path):
    f = _write_yaml(
        tmp_path,
        """
        vexel:
          store:
            backend: pinecone
            pinecone:
              api_key: test-key
              index_name: my-index
        """,
    )
    cfg = VexelConfig.from_yaml(f)
    assert cfg.store.backend == "pinecone"
    assert cfg.store.pinecone.api_key == "test-key"
    assert cfg.store.pinecone.index_name == "my-index"