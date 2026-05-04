# vexel: Open-Source Visual Search Library

## Complete Design Document

---

# 1. Executive Summary

**vexel** is a domain-agnostic Python library for building multi-vector image search systems. It solves the core technical problems — image preprocessing, multi-representation indexing, deduplication, placeholder filtering, and score-gap thresholding — leaving business logic (catalog hydration, filtering rules) to the integrator.

The library is built for e-commerce, healthcare, and any catalog where a single product has multiple visual representations (box, blister strip, loose item; main shot, side view, sole; etc.).

**Target Users:** ML engineers, backend engineers, data engineers building search features.

**Core Value:** One indexing pipeline works for medicines, sneakers, furniture, or spare parts. Just configure it via YAML.

---

# 2. The Boundary Decision

## Open Source: vexel

Everything independent of your catalog, database, and business rules.

| Component | Rationale |
|---|---|
| Image preprocessing (crop, pad, normalize) | Pure CV, zero domain knowledge |
| Multi-vector indexer with composite UUID | Universal: one entity, many images |
| Encoder abstraction (CLIP, SigLIP, custom) | Model choice varies by deployment |
| Vector store adapters (Qdrant, Pinecone, Weaviate) | Infrastructure varies by team |
| pHash placeholder registry + filtering | Universal data quality problem |
| Entity-level deduplication + score-gap thresholding | Any multi-vector system needs this |
| Config schema (Pydantic) | YAML-driven, no code changes per use case |

## Implementation Layer: Your Service

Everything that touches your business domain.

| Component | Rationale |
|---|---|
| ES/SKU hydration | Tied to your data model |
| saleable, is_discontinued, city filters | Business rules, not search logic |
| is_rx, category payload fields | Domain-specific metadata |
| primary_box / blister_strip / loose_pill labels | Domain vocabulary (configured in YAML) |
| API endpoints, auth, rate limiting | Infrastructure choice |
| Analytics, GA, RudderStack | Deployment-specific |

---

# 3. Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                     vexel  (open source)                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  image_pipeline                                             │ │
│  │   ┌──────────────┐  ┌────────────────┐  ┌───────────────┐   │ │
│  │   │ ImageFetcher │→ │  Preprocessor  │→ │  PhashFilter  │   │ │
│  │   │(CDN transform│  │(crop, pad,     │  │(registry load,│   │ │
│  │   │ pluggable fn)│  │ bg remove,     │  │ freq detect,  │   │ │
│  │   └──────────────┘  │ square pad)    │  │ query filter) │   │ │
│  │                     └────────────────┘  └───────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  encoder                                                    │ │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │ │
│  │   │ CLIPEncoder │   │SigLIPEncoder│   │  BaseEncoder    │   │ │
│  │   │(ViT-B/32,   │   │(future)     │   │  (custom)       │   │ │
│  │   │ ViT-L/14)   │   └─────────────┘   └─────────────────┘   │ │
│  │   └─────────────┘                                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│              ┌────────────────┬────────────────┐                 │
│              ▼                ▼                ▼                 │
│  ┌──────────────────┐  ┌───────────────────────────────┐         │
│  │ indexer          │  │ search                        │         │
│  │ MultiVectorIndex │  │ VisualSearchEngine            │         │
│  │ - composite UUID │  │ - encode query                │         │
│  │ - image mapping  │  │ - ANN search (over-fetch)     │         │
│  │ - concurrent     │  │ - entity dedup                │         │
│  │ - batch upsert   │  │ - score-gap threshold         │         │
│  └────────┬─────────┘  └────────┬──────────────────────┘         │
│           │                     │                                │
│           └──────────┬──────────┘                                │
│                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ store  (adapters)                                           │ │
│  │ BaseVectorStore (ABC)                                       │ │
│  │ ├─ QdrantAdapter                                            │ │
│  │ ├─ PineconeAdapter (stub)                                   │ │
│  │ └─ WeaviateAdapter (stub)                                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ config                                                      │ │
│  │ vexelConfig (Pydantic) — YAML schema + validation           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                               ▲
                  implements / plugs into
                               │
┌──────────────────────────────────────────────────────────────────┐
│            Your Implementation Service                           │
│                                                                  │
│  EntitySource        MetadataStore       BusinessFilters         │
│  (ES scroll,    →    (hydrate from   →   ()                      │
│   DB, CSV)           any DB)                                     │
│                                                                  │
│  ImageTypeConfig     CDNTransformFn      AnalyticsHook           │
│  (your labels)       (Gumlet, etc.)      (GA/RudderStack)        │
└──────────────────────────────────────────────────────────────────┘
```

---

# 4. Package Structure

```text
vexel/
├── vexel/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── image_pipeline/
│   │   ├── __init__.py
│   │   ├── fetcher.py
│   │   ├── preprocessor.py
│   │   └── phash_filter.py
│   │
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── clip.py
│   │   └── factory.py
│   │
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── multi_vector.py
│   │   ├── uuid_utils.py
│   │   └── image_type_config.py
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── threshold.py
│   │
│   └── store/
│       ├── __init__.py
│       ├── base.py
│       ├── qdrant.py
│       ├── pinecone.py
│       └── weaviate.py
│
├── examples/
├── tests/
├── pyproject.toml
└── README.md
```

---

# 5. Configuration Contract

```yaml
# vexel.yaml

vexel:
  encoder:
    name: clip
    model: ViT-B/32
    device: cpu
    batch_size: 8

  store:
    backend: qdrant
    qdrant:
      host: localhost
      port: 6333
      collection: entity_images
      vector_size: 512

  pipeline:
    preprocessor:
      enabled: true
      target_size: 224
      padding_color: [128, 128, 128]
      bg_threshold: 15
      margin_pct: 0.05

    cdn_transform:
      enabled: true
      template: "{url}?w={size}&h={size}&bg=white"

  placeholder_filter:
    registry_path: ./placeholder_hashes.json
    frequency_threshold: 3

  indexer:
    max_images_per_entity: 3
    concurrent_workers: 4
    batch_upsert_size: 100
    namespace_uuid: "12345678-1234-5678-1234-567812345678"

    image_type_mapping:
      0: primary
      1: secondary
      2: tertiary

  search:
    over_fetch_multiplier: 3
    score_gap_threshold: 0.08
    min_score: 0.70
    max_results: 20
```

---

# 6. Component Specifications

## 6.1 MultiVectorIndexer

```python
from vexel import MultiVectorIndexer, vexelConfig

config = vexelConfig.from_yaml("vexel.yaml")
indexer = MultiVectorIndexer(config)

async def get_image_urls(entity: dict) -> list[str]:
    return entity.get("image_urls", [])[:config.indexer.max_images_per_entity]

await indexer.run(
    entities=my_es_scroll(),
    entity_id_fn=lambda e: str(e["id"]),
    image_urls_fn=get_image_urls,
    extra_payload_fn=lambda e: {
        "category": e.get("category"),
    },
)
```

### Internally

1. Get image URLs
2. Apply CDN transform
3. Download concurrently
4. Preprocess images
5. pHash filtering
6. Encode through BaseEncoder
7. Generate composite UUID
8. Batch upsert
9. Update placeholder registry

---

## 6.2 VisualSearchEngine

```python
from vexel import VisualSearchEngine, vexelConfig

engine = VisualSearchEngine(config)

results = await engine.search(
    image_bytes=raw_image_bytes,
    top_k=20,
)
```

### Internally

1. Preprocess query image
2. Encode query
3. ANN search with over-fetch
4. Entity deduplication
5. Score-gap thresholding
6. Return `SearchCandidate`

---

## 6.3 Preprocessor

Pipeline:

```text
Input bytes
  → PIL.open + RGB conversion
  → Alpha channel check
  → Background detection
  → Contour detection
  → Crop
  → Pad to square
  → Resize
  → Return processed image
```

---

## 6.4 BaseVectorStore

```python
class BaseVectorStore(ABC):
    async def connect(self) -> None: ...
    async def upsert(self, points: list[VectorPoint]) -> None: ...
    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filter_payload: dict | None = None,
    ) -> list[RawHit]: ...
    async def close(self) -> None: ...
```

---

## 6.5 BaseEncoder

```python
class BaseEncoder(ABC):
    async def load(self) -> None: ...
    async def encode(self, image_bytes: bytes) -> list[float]: ...
    async def encode_batch(self, images: list[bytes]) -> list[list[float]]: ...
    async def close(self) -> None: ...
```

---

# 7. Data Flow: Indexing

```text
EntitySource
    ↓
MultiVectorIndexer
    ↓
Image Fetching
    ↓
Preprocessing
    ↓
pHash Filtering
    ↓
Encoding
    ↓
UUID Generation
    ↓
Vector Store Upsert
```

---

# 8. Data Flow: Search

```text
HTTP Request (image)
    ↓
VisualSearchEngine
    ↓
Preprocess
    ↓
Encode
    ↓
Vector Search
    ↓
Deduplicate
    ↓
Thresholding
    ↓
SearchCandidate[]
    ↓
Your Service
```

---

# 9. System Design Decisions & Tradeoffs

## D1: Why Not Include the HTTP Layer?

Framework choice should remain implementation-specific.

---

## D2: Why Is ImageTypeConfig in YAML?

Avoids domain lock-in.

---

## D3: Query and Index Pipelines Must Match

Identical preprocessing ensures embedding consistency.

---

## D4: Score-Gap Thresholding

More reliable than fixed thresholds alone.

---

## D5: Composite UUID Strategy

```python
uuid5(namespace, f"{entity_id}::{image_type}")
```

Prevents overwrite collisions.

---

## D6: pHash Registry + Payload

Both runtime filtering and auditability are needed.

---

# 10. Integration Example

## `vexel.yaml`

```yaml
vexel:
  encoder:
    name: clip
    model: ViT-B/32
    device: cuda

  store:
    backend: qdrant

  indexer:
    max_images_per_entity: 3

    image_type_mapping:
      0: main_shot
      1: side_view
      2: sole_detail
```

---

## `indexer_job.py`

```python
import asyncio
from vexel import MultiVectorIndexer, vexelConfig

async def main():
    config = vexelConfig.from_yaml("vexel.yaml")
    indexer = MultiVectorIndexer(config)

    await indexer.run(
        entities=scroll_products(),
        entity_id_fn=lambda p: p["sku"],
        image_urls_fn=lambda p: p["images"],
    )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## `api.py`

```python
from fastapi import FastAPI, UploadFile, File
from vexel import VisualSearchEngine, vexelConfig

config = vexelConfig.from_yaml("vexel.yaml")
engine = VisualSearchEngine(config)

app = FastAPI()

@app.post("/visual-search")
async def visual_search(image: UploadFile = File(...)):
    candidates = await engine.search(await image.read(), top_k=20)
    return candidates
```

---

# 11. Framework Choice

| Layer | Choice |
|---|---|
| Core Library | asyncio + aiohttp |
| Reference Server | FastAPI |
| CLI | typer |
| Your Service | Any framework |

---

# 12. What Stays in Your Implementation

| File | Reason |
|---|---|
| sku_hydrator.py | Your schema |
| visual_search_manager.py | Business rules |
| listeners.py | Framework-specific |
| config.json | Deployment config |
| seed_visual_search.py | ES structure |
| Image type labels | Domain vocabulary |

---

# 13. Dependency Management

```toml
[project.optional-dependencies]
clip = ["open-clip-torch>=2.20"]
qdrant = ["qdrant-client>=1.9"]
pinecone = ["pinecone-client>=3.0"]
opencv = ["opencv-python-headless>=4.8"]
```

### Installation

```bash
pip install vexel
pip install vexel[clip,qdrant]
pip install vexel[clip,qdrant,opencv]
```

---

# 14. Getting Started

```bash
# Install
pip install vexel[clip,qdrant]

# Create config
cp examples/pharmacy_config.yaml vexel.yaml

# Run indexer
python indexer_job.py

# Start server
uvicorn api:app --reload

# Test
curl -F "image=@product.jpg" http://localhost:8000/visual-search
```

---

# 15. Success Metrics

| Metric | Target |
|---|---|
| Time to first integration | < 1 hour |
| Lines of user code | < 100 |
| Preprocessing latency | < 15ms/image |
| Search latency | < 200ms |
| Recall@10 improvement | +3–5% |
| Disk footprint | < 5GB |

---

# 16. Roadmap (Phase 2+)

- Saliency-based cropping
- SigLIPEncoder
- Weaviate + Pinecone completion
- Fine-tuning recipes (see [PharmaCLIP training guide](results/pharmaclip_training.md))
- Batch query API
- Analytics hooks
- Streaming indexer

---

# 17. Production Cost & Scaling

> Numbers derived from measured benchmarks on Apple M-series CPU and extrapolated to AWS instance types.
> See [`results/flipkart_fashion_eval.md`](results/flipkart_fashion_eval.md) for the underlying benchmark.

## 17.1 Per-query latency breakdown

```
CLIP encode (ViT-B/32, CPU)    :  18–20 ms   ← dominant bottleneck
Qdrant HNSW ANN (300k vectors) :   2–4 ms
Dedup + filter + sort          :    ~1 ms
──────────────────────────────────────────
Total per query (CPU)          :  ~22–25 ms
```

## 17.2 Indexing 100k SKUs (one-time batch job)

| Config | Vectors |
|---|---|
| 1 image / SKU | 100,000 |
| 3 images / SKU (box + strip + pill) | 300,000 |

| Instance | CLIP encode / img | Throughput | Time (300k imgs) | Spot cost | Total cost |
|---|---|---|---|---|---|
| c5.2xlarge (8 vCPU, CPU) | ~50 ms | 160 img/s | ~31 min | $0.034/hr | **< $0.02** |
| g4dn.xlarge (T4, single) | ~5 ms | 200 img/s | ~25 min | $0.158/hr | **< $0.07** |
| g4dn.xlarge (T4, batch=32) | ~12 ms | 2,700 img/s | **~2 min** | $0.158/hr | **< $0.01** |

Indexing is a negligible cost. A full re-index of 100k SKUs (3 images each) completes in under 2 minutes on a single spot T4.

### Qdrant storage for 300k vectors

```
300k × 512-d float32          =  614 MB raw vectors
HNSW graph overhead (~2-3×)   ≈  1.2–1.8 GB
Payload metadata               ≈  200 MB
─────────────────────────────────────────────
Total RAM required             ≈  2 GB
```

A `r6i.small` (2 vCPU, 4 GB RAM) at $0.063/hr = **$46/month** comfortably handles up to 300k vectors with room for HNSW rebuilds.

## 17.3 Serving throughput and monthly cost

| Scale | Searches / day | Typical for |
|---|---|---|
| 10 QPS | 864k | Mid-size app |
| 100 QPS | 8.6M | Large marketplace (peak) |
| 1,000 QPS | 86M | Amazon / Flipkart full peak |

### Max QPS per instance

| Instance | CLIP encode | Max QPS (8 workers) |
|---|---|---|
| c5.2xlarge (8 vCPU, CPU) | ~50 ms | ~160 |
| c5.4xlarge (16 vCPU, CPU) | ~50 ms | ~320 |
| g4dn.xlarge (T4, batch=1) | ~5 ms | ~800 |
| g4dn.xlarge (T4, batch=16) | ~12 ms total | **~1,300** |

### Monthly infrastructure cost

| Scale | Architecture | Monthly cost |
|---|---|---|
| 5–10 QPS (today) | 1× c5.2xlarge + r6i.small | **~$100** |
| 100 QPS | 1× g4dn.xlarge + r6i.large | **~$450** |
| 1,000 QPS (on-demand) | 2× g4dn.xlarge + r6i.large | **~$910** |
| 1,000 QPS (spot fleet) | 4× g4dn.xlarge spot + r6i.large | **~$380** |
| 1,000 QPS (1-yr reserved) | 2× g4dn.xlarge reserved + r6i | **~$530** |

## 17.4 Critical scaling path

The current encoder processes one image per call. GPU utilisation is poor at low batch sizes. The fix is **dynamic request batching** — collect 16–32 requests over a 10–20 ms window and run a single forward pass:

```
Single image on T4:    5 ms  →   200 QPS per GPU
Batch of 16 on T4:    12 ms  →  1,333 QPS per GPU  (6.7× gain)
```

The `BaseEncoder` ABC already declares `encode_batch()`; only the `CLIPEncoder` implementation needs this method added before GPU deployment makes sense.

---

# Final Philosophy

> This is vexel: a library so simple that integrating it takes less time than reading the docs.