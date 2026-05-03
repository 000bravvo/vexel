"""
Flipkart Fashion — Visual Search Integration Evaluation
=======================================================

Cross-view self-retrieval test using the Kaggle "Flipkart Fashion Products"
dataset (30 k products, 2 Flipkart CDN images each).

**What it measures**
  Index every product using its PRIMARY image (images[0]).
  Query 25 % of those products using their SECONDARY image (images[1]).
  A hit = the ground-truth product ID appears in the returned candidates.

  Metrics reported:
    Recall@1 / @5 / @10  — fraction of queries where GT rank ≤ k
    MRR@10               — mean reciprocal rank
    Avg top-1 score      — mean cosine similarity of the top result

**Dataset download (one-time)**
    import kagglehub
    path = kagglehub.dataset_download("aaditshukla/flipkart-fasion-products-dataset")

**Run**
    python tests/integration/flipkart_fashion_eval.py
    python tests/integration/flipkart_fashion_eval.py --index-size 5000 --top-k 10
    python tests/integration/flipkart_fashion_eval.py --index-size 0  # all 30 k

**Requirements**
    - Qdrant running on localhost:6333
    - vexel installed with [clip,qdrant] extras
    - kagglehub installed (pip install kagglehub)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("flipkart_eval")

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

_DEFAULT_DATASET_PATH = (
    Path.home()
    / ".cache/kagglehub/datasets/aaditshukla"
    / "flipkart-fasion-products-dataset/versions/3"
    / "flipkart_fashion_products_dataset.json"
)

_COLLECTION = "flipkart_fashion_eval"
# Fresh UUID5 namespace — isolated from the pharmacy sku_images collection.
_NAMESPACE_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def upgrade_image_url(url: str) -> str:
    """Rewrite Flipkart CDN thumbnail URL from 128×128 → 512×512."""
    return re.sub(r"/image/\d+/\d+/", "/image/512/512/", url)


def load_dataset(path: str | Path, min_images: int = 2) -> list[dict]:
    """Load dataset JSON; keep only products with enough image URLs."""
    data = json.loads(Path(path).read_text())
    filtered = [p for p in data if len(p.get("images", [])) >= min_images]
    logger.info(
        "Loaded %d products (of %d total) with ≥%d images",
        len(filtered),
        len(data),
        min_images,
    )
    return filtered


async def _download_image(url: str, timeout: int = 10) -> bytes:
    """Download image bytes (blocking I/O executed in thread pool)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "vexel-eval/0.1"}
    )
    loop = asyncio.get_event_loop()

    def _fetch() -> bytes:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    return await loop.run_in_executor(None, _fetch)


# ------------------------------------------------------------------ #
# Single query worker                                                  #
# ------------------------------------------------------------------ #


async def _query_one(
    product: dict,
    engine: Any,
    top_k: int,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Download secondary image, run search, return result dict."""
    pid = product["pid"]
    secondary_url = upgrade_image_url(product["images"][1])

    async with sem:
        # Download secondary image
        try:
            image_bytes = await _download_image(secondary_url)
        except Exception as exc:
            logger.debug("Download failed for %s: %s", pid, exc)
            return None

        # Search
        try:
            from vexel.search.engine import VisualSearchEngine  # noqa: F401
            candidates = await engine.search(image_bytes, top_k=top_k)
        except Exception as exc:
            logger.debug("Search failed for %s: %s", pid, exc)
            return None

    # Find ground-truth rank
    rank: int | None = None
    for r, c in enumerate(candidates, start=1):
        if c.entity_id == pid:
            rank = r
            break

    return {
        "pid": pid,
        "title": product.get("title", ""),
        "brand": product.get("brand", ""),
        "sub_category": product.get("sub_category", ""),
        "rank": rank,
        "n_candidates": len(candidates),
        "top1_score": candidates[0].score if candidates else 0.0,
        "top1_entity_id": candidates[0].entity_id if candidates else None,
        "top5": [
            {
                "entity_id": c.entity_id,
                "score": round(c.score, 4),
                "image_type": c.matched_image_type,
            }
            for c in candidates[:5]
        ],
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flipkart Fashion visual search evaluation using vexel"
    )
    parser.add_argument(
        "--index-size",
        type=int,
        default=2000,
        help="Products to index (0 = all; default: 2000)",
    )
    parser.add_argument(
        "--query-fraction",
        type=float,
        default=0.25,
        help="Fraction of indexed products used as queries (default: 0.25)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Candidates to retrieve per query"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(_DEFAULT_DATASET_PATH),
        help="Path to flipkart_fashion_products_dataset.json",
    )
    parser.add_argument(
        "--qdrant-host", type=str, default="localhost"
    )
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel query workers (default: 8)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Load dataset ────────────────────────────────────────────
    products = load_dataset(args.dataset_path, min_images=2)

    if args.index_size > 0 and args.index_size < len(products):
        products = random.sample(products, args.index_size)

    logger.info("Index set: %d products", len(products))

    # ── 2. Configure vexel ─────────────────────────────────────────
    from vexel.config import (
        CDNTransformConfig,
        EncoderConfig,
        IndexerConfig,
        PipelineConfig,
        PreprocessorConfig,
        QdrantStoreConfig,
        SearchConfig,
        StoreConfig,
        VexelConfig,
    )
    from vexel.encoder.clip import CLIPEncoder
    from vexel.indexer.multi_vector import MultiVectorIndexer
    from vexel.search.engine import VisualSearchEngine
    from vexel.store.qdrant import QdrantAdapter

    config = VexelConfig(
        encoder=EncoderConfig(
            model="ViT-B/32",
            pretrained="openai",
            device="cpu",
        ),
        store=StoreConfig(
            backend="qdrant",
            qdrant=QdrantStoreConfig(
                host=args.qdrant_host,
                port=args.qdrant_port,
                collection=_COLLECTION,
                vector_size=512,
            ),
        ),
        pipeline=PipelineConfig(
            # Disable CDN transform — Flipkart uses path-based sizing.
            # We upgrade URLs ourselves in image_urls_fn.
            cdn_transform=CDNTransformConfig(enabled=False),
            # Disable smart-crop: thumbnails are already well-cropped.
            # This also ensures index and query vectors use the same
            # encode path (raw bytes → CLIP resize → embedding).
            preprocessor=PreprocessorConfig(
                enabled=False,
                download_size=512,
                target_size=224,
            ),
        ),
        indexer=IndexerConfig(
            max_images_per_entity=1,          # only index primary image
            concurrent_workers=8,
            batch_upsert_size=100,
            namespace_uuid=_NAMESPACE_UUID,
            image_type_mapping={0: "primary"},
        ),
        search=SearchConfig(
            over_fetch_multiplier=3,
            score_gap_threshold=0.15,
            min_score=0.50,                   # cross-view fashion needs relaxed threshold
            max_results=20,
        ),
    )

    # ── 3. Init encoder & store ────────────────────────────────────
    encoder = CLIPEncoder(config.encoder)
    await encoder.load()
    logger.info("CLIP encoder ready (ViT-B/32 openai)")

    # Drop + recreate collection for a reproducible run
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams

    qdrant_raw = AsyncQdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    existing = {c.name for c in (await qdrant_raw.get_collections()).collections}
    if _COLLECTION in existing:
        await qdrant_raw.delete_collection(_COLLECTION)
        logger.info("Dropped existing collection '%s'", _COLLECTION)
    await qdrant_raw.create_collection(
        collection_name=_COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    logger.info("Created fresh collection '%s' (512-d cosine)", _COLLECTION)
    await qdrant_raw.close()

    store = QdrantAdapter(config.store.qdrant)
    await store.connect()

    # ── 4. Index primary images ────────────────────────────────────
    indexer = MultiVectorIndexer(config, encoder, store)
    logger.info("Indexing %d products (primary image, 512×512)…", len(products))
    t_index_start = time.perf_counter()

    index_run = await indexer.run(
        entities=products,
        entity_id_fn=lambda p: p["pid"],
        image_urls_fn=lambda p: [upgrade_image_url(p["images"][0])],
        extra_payload_fn=lambda p: {
            "title": p.get("title", ""),
            "brand": p.get("brand", ""),
            "category": p.get("category", ""),
            "sub_category": p.get("sub_category", ""),
        },
    )

    index_elapsed = time.perf_counter() - t_index_start
    logger.info(
        "Indexing done in %.1f s — %d vectors upserted, "
        "%d blank-skipped, %d failed",
        index_elapsed,
        index_run.indexed_vectors,
        index_run.skipped_blank,
        index_run.failed_entities,
    )

    # ── 5. Sample 25 % as queries ──────────────────────────────────
    n_queries = max(1, int(len(products) * args.query_fraction))
    query_products = random.sample(products, n_queries)
    logger.info(
        "Query set: %d products (%.0f %% of index), secondary image",
        n_queries,
        args.query_fraction * 100,
    )

    # ── 6. Run queries in parallel ─────────────────────────────────
    engine = VisualSearchEngine(config, encoder, store)
    sem = asyncio.Semaphore(args.concurrency)

    t_query_start = time.perf_counter()
    raw_results = await asyncio.gather(
        *[_query_one(p, engine, args.top_k, sem) for p in query_products]
    )
    query_elapsed = time.perf_counter() - t_query_start

    results = [r for r in raw_results if r is not None]
    logger.info(
        "Queries done in %.1f s — %d/%d successful (%.3f s/query)",
        query_elapsed,
        len(results),
        n_queries,
        query_elapsed / max(len(results), 1),
    )

    # ── 7. Compute metrics ─────────────────────────────────────────
    n = len(results)
    if n == 0:
        logger.error("No successful query results — cannot compute metrics.")
        return

    hits_at_1  = [r for r in results if r["rank"] == 1]
    hits_at_5  = [r for r in results if r["rank"] is not None and r["rank"] <= 5]
    hits_at_10 = [r for r in results if r["rank"] is not None and r["rank"] <= 10]
    reciprocal_ranks = [1.0 / r["rank"] for r in results if r["rank"] is not None]

    recall_1  = len(hits_at_1) / n
    recall_5  = len(hits_at_5) / n
    recall_10 = len(hits_at_10) / n
    mrr       = sum(reciprocal_ranks) / n
    avg_top1  = sum(r["top1_score"] for r in results) / n
    avg_rank  = (
        sum(r["rank"] for r in results if r["rank"] is not None)
        / max(len(reciprocal_ranks), 1)
    )

    # ── 8. Report ──────────────────────────────────────────────────
    bar = "=" * 66

    print(f"\n{bar}")
    print("  FLIPKART FASHION  ·  VISUAL SEARCH EVALUATION (cross-view)")
    print(bar)
    print(f"  Dataset           : Kaggle flipkart-fasion-products-dataset")
    print(f"  Index set         : {len(products):,} products  ({index_run.indexed_vectors:,} vectors)")
    print(f"  Query set         : {n:,} products  (25 % of index, secondary image)")
    print(f"  Model             : CLIP ViT-B/32  (openai weights, CPU)")
    print(f"  Collection        : {_COLLECTION}")
    print(f"  Top-K             : {args.top_k}")
    print(f"  Index time        : {index_elapsed:.1f} s  ({index_elapsed / len(products):.2f} s/product)")
    print(f"  Query time        : {query_elapsed:.1f} s  ({query_elapsed / n:.3f} s/query)")
    print(f"  Failed / skipped  : index={index_run.failed_entities}  queries={n_queries - n}")
    print(f"  {'─'*62}")
    print(f"  Recall@1          : {recall_1:.3f}   ({len(hits_at_1):,}/{n:,})")
    print(f"  Recall@5          : {recall_5:.3f}   ({len(hits_at_5):,}/{n:,})")
    print(f"  Recall@10         : {recall_10:.3f}   ({len(hits_at_10):,}/{n:,})")
    print(f"  MRR@10            : {mrr:.3f}")
    print(f"  Avg top-1 score   : {avg_top1:.4f}")
    print(f"  Avg hit rank      : {avg_rank:.1f}  (of hits only)")
    print(bar)

    # Per-category recall
    cat_counts: dict[str, list[bool]] = {}
    for r in results:
        cat = r.get("sub_category") or "Unknown"
        cat_counts.setdefault(cat, []).append(r["rank"] is not None and r["rank"] <= 10)

    if cat_counts:
        print("\n  RECALL@10 BY SUB-CATEGORY  (top-8 most queried):\n")
        sorted_cats = sorted(cat_counts.items(), key=lambda x: -len(x[1]))[:8]
        for cat, hits_list in sorted_cats:
            cat_r10 = sum(hits_list) / len(hits_list)
            bar_len = int(cat_r10 * 30)
            bar_str = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {cat:<22}  {bar_str}  {cat_r10:.2f}  (n={len(hits_list):>4})")

    # Qualitative examples: 5 hits + 5 misses
    hits   = [r for r in results if r["rank"] is not None]
    misses = [r for r in results if r["rank"] is None]

    sample_hits   = random.sample(hits,   min(5, len(hits)))
    sample_misses = random.sample(misses, min(5, len(misses)))

    print(f"\n  SAMPLE HITS (found in top-{args.top_k}):\n")
    for r in sample_hits:
        print(f"  ✓ rank {r['rank']:>2} | {r['pid']}  score={r['top5'][r['rank']-1]['score']:.4f}")
        print(f"         {r['title'][:58]}")
        print(f"         [{r['sub_category']}  ·  {r['brand']}]")
        for i, c in enumerate(r["top5"][:3], 1):
            arrow = "→" if c["entity_id"] == r["pid"] else " "
            print(f"         {arrow} #{i}  {c['entity_id']}  {c['score']:.4f}")
        print()

    if sample_misses:
        print(f"  SAMPLE MISSES (not found in top-{args.top_k}):\n")
        for r in sample_misses:
            print(f"  ✗ miss  | {r['pid']}")
            print(f"         {r['title'][:58]}")
            print(f"         [{r['sub_category']}  ·  {r['brand']}]")
            print(f"         top-1: {r['top1_entity_id']}  score={r['top1_score']:.4f}")
            print()

    print(bar)

    # ── 9. Cleanup ─────────────────────────────────────────────────
    indexer.close()
    await encoder.close()
    await store.close()
    logger.info("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())