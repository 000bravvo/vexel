"""
CLIPEncoder — open_clip-based image encoder.

Implements BaseEncoder using the open_clip library.

Supported models: any open_clip model name (ViT-B/32, ViT-L/14, etc.)
Pretrained weights: "openai", "laion400m", "laion2b", etc.

The model is loaded synchronously inside a thread-pool executor so the
asyncio event loop is never blocked during the ~3 s cold start.

Pre-processing pipeline (identical to the offline indexer to preserve the
critical invariant that index and query transforms must match):

    open_clip preprocess_val:
        Resize(256, antialias) → CenterCrop(224) → ToTensor
        → Normalize(CLIP ImageNet stats)

If the preprocessed image is already a PIL Image (e.g. from Preprocessor),
the encoder skips the bytes-decode step.
"""

from __future__ import annotations

import asyncio
import io
import logging

from vexel.config import EncoderConfig
from vexel.encoder.base import BaseEncoder

logger = logging.getLogger(__name__)


class CLIPEncoder(BaseEncoder):
    """
    Encoder wrapping open_clip (ViT-B/32 by default).

    Requires the optional dependency: pip install vexel[clip]

    Example::

        encoder = CLIPEncoder(config.encoder)
        await encoder.load()
        vector = await encoder.encode(image_bytes)   # list[float], len=512
    """

    def __init__(self, config: EncoderConfig) -> None:
        self._config = config
        self._model = None
        self._preprocess = None
        self._device: str = config.device

    # ---------------------------------------------------------------- #
    # Lifecycle                                                          #
    # ---------------------------------------------------------------- #

    def _load_sync(self) -> None:
        """Blocking model load — runs in executor, never on the event loop."""
        import open_clip

        # Normalise slash-style model names ("ViT-B/32" → "ViT-B-32")
        model_name = self._config.model
        pretrained = getattr(self._config, "pretrained", "openai")

        logger.info(
            "Loading CLIP model %s / %s on %s", model_name, pretrained, self._device
        )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("CLIP model ready")

    async def load(self) -> None:
        """Load the model in a thread-pool executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

    async def close(self) -> None:
        """Release model memory."""
        self._model = None
        self._preprocess = None
        logger.info("CLIP encoder released")

    # ---------------------------------------------------------------- #
    # Encoding                                                           #
    # ---------------------------------------------------------------- #

    def _encode_pil_sync(self, pil_image) -> list[float]:
        """Blocking encode of a PIL Image — runs in executor."""
        import torch

        tensor = self._preprocess(pil_image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def _encode_bytes_sync(self, image_bytes: bytes) -> list[float]:
        """Blocking encode of raw image bytes — runs in executor."""
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._encode_pil_sync(img)

    async def encode(self, image_bytes: bytes) -> list[float]:
        """
        Encode raw image bytes to an L2-normalised float32 vector.

        Runs in a thread executor so the event loop is not blocked.

        Args:
            image_bytes: raw bytes of a JPEG / PNG / WebP image.

        Returns:
            list[float] of length matching the model's embedding dimension
            (512 for ViT-B/32, 768 for ViT-L/14), L2-normalised.

        Raises:
            RuntimeError: if called before load().
        """
        if not self.is_ready:
            raise RuntimeError(
                "CLIPEncoder.encode() called before load()"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._encode_bytes_sync, image_bytes
        )

    async def encode_pil(self, pil_image) -> list[float]:
        """
        Encode a PIL Image directly, skipping the bytes-decode step.

        Used by the indexer when Preprocessor has already produced a PIL
        Image — avoids a redundant encode→decode round-trip.
        """
        if not self.is_ready:
            raise RuntimeError("CLIPEncoder.encode_pil() called before load()")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._encode_pil_sync, pil_image
        )

    def encode_pil_sync(self, pil_image) -> list[float]:
        """
        Synchronous PIL-image encode for use inside ThreadPoolExecutor workers.

        This method is intentionally synchronous: the multi-vector indexer
        runs encode inside a ThreadPoolExecutor thread (not on the event loop)
        because torch releases the GIL during forward passes, giving real
        parallelism across CPU workers.
        """
        if not self.is_ready:
            raise RuntimeError(
                "CLIPEncoder.encode_pil_sync() called before load()"
            )
        return self._encode_pil_sync(pil_image)

    # ---------------------------------------------------------------- #
    # Introspection                                                      #
    # ---------------------------------------------------------------- #

    @property
    def is_ready(self) -> bool:
        return self._model is not None