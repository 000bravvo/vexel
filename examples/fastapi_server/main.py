"""
Reference FastAPI server — Visual Search API using Vexel.

This file shows how to integrate Vexel into any async Python web framework.
It is intentionally minimal so you can adapt it to your own project.

Run (development):
    pip install fastapi uvicorn vexel[clip,qdrant]
    CONFIG=examples/pharmacy_config.yaml uvicorn examples.fastapi_server.main:app --reload

Endpoints:
    POST /search           — search by image upload
    GET  /health           — liveness check
    GET  /docs             — Swagger UI (FastAPI built-in)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from vexel import SearchCandidate, VisualSearchEngine, VexelConfig
from vexel.encoder.factory import get_encoder
from vexel.store.factory import get_store

logger = logging.getLogger("vexel.example")

# ------------------------------------------------------------------ #
# App startup / shutdown                                               #
# ------------------------------------------------------------------ #

_engine: VisualSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once on startup; release on shutdown."""
    global _engine

    config_path = os.environ.get("CONFIG", "vexel.yaml")
    logger.info("Loading Vexel config from %s", config_path)
    config = VexelConfig.from_yaml(config_path)

    encoder = get_encoder(config.encoder)
    await encoder.load()
    logger.info("Encoder loaded: %s", config.encoder.model)

    store = get_store(config.store)
    await store.connect()
    logger.info("Vector store connected: %s", config.store.backend)

    _engine = VisualSearchEngine(config, encoder, store)
    logger.info("VisualSearchEngine ready")

    yield  # app runs here

    # --- Shutdown ---
    await encoder.close()
    await store.close()
    logger.info("Resources released")


app = FastAPI(
    title="Vexel Visual Search",
    description="Image-to-entity similarity search powered by Vexel.",
    version="0.1.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "engine_ready": _engine is not None}


@app.post("/search")
async def search_by_image(
    file: UploadFile = File(..., description="Query image (JPEG / PNG / WebP)"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results to return"),
):
    """
    Upload an image and receive the most visually similar entities.

    Returns a ranked list of matches with entity IDs and scores.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        candidates: list[SearchCandidate] = await _engine.search(
            image_bytes=image_bytes,
            top_k=top_k,
        )
    except Exception as exc:
        logger.exception("Search error: %s", exc)
        raise HTTPException(status_code=500, detail="Search failed") from exc

    return JSONResponse(
        content={
            "query_top_k": top_k,
            "result_count": len(candidates),
            "results": [
                {
                    "entity_id": c.entity_id,
                    "score": c.score,
                    "matched_image_type": c.matched_image_type,
                }
                for c in candidates
            ],
        }
    )