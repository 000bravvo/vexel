"""
BaseEncoder — abstract interface for image embedding models.

Implement this to add SigLIP, BLIP, a fine-tuned model, or an API-based
encoder (e.g. OpenAI Embeddings) without changing any indexer or search code.

Design notes:
  - load() / close() are async so implementations can use aiohttp for
    API-based encoders, or asyncio.run_in_executor for CPU-heavy models.
  - encode() accepts bytes so the caller never needs to decode twice.
  - encode_batch() enables GPU-friendly batching; the default
    implementation simply loops over encode(), which is correct for
    CPU-only encoders but should be overridden for GPU efficiency.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """
    Abstract base class for image-to-vector encoders.

    Subclass and implement load(), encode(), and optionally encode_batch().
    close() defaults to a no-op; override if you need teardown.

    Example::

        class MyEncoder(BaseEncoder):
            async def load(self) -> None:
                self._model = load_my_model()

            async def encode(self, image_bytes: bytes) -> list[float]:
                img = decode(image_bytes)
                return self._model.infer(img)
    """

    # ---------------------------------------------------------------- #
    # Lifecycle                                                          #
    # ---------------------------------------------------------------- #

    @abstractmethod
    async def load(self) -> None:
        """
        Initialise the encoder (load model weights, allocate GPU memory, etc.).

        Called once at startup.  Should be idempotent.
        """

    async def close(self) -> None:
        """
        Release resources held by the encoder.

        Default is a no-op; override if teardown is needed.
        """

    # ---------------------------------------------------------------- #
    # Encoding                                                           #
    # ---------------------------------------------------------------- #

    @abstractmethod
    async def encode(self, image_bytes: bytes) -> list[float]:
        """
        Encode a single image to an L2-normalised embedding vector.

        Args:
            image_bytes: raw bytes of a JPEG / PNG / WebP image,
                         OR bytes from a preprocessed PIL Image.

        Returns:
            L2-normalised list[float].  Length determined by the model
            (e.g. 512 for ViT-B/32, 768 for ViT-L/14).
        """

    async def encode_batch(self, images: list[bytes]) -> list[list[float]]:
        """
        Encode a list of images in one batch.

        Default: sequential encode() calls.  Override for GPU batch inference
        where a single forward pass over N images is much faster than N passes.

        Args:
            images: list of raw image bytes.

        Returns:
            list of L2-normalised embedding vectors.
        """
        return [await self.encode(img) for img in images]

    # ---------------------------------------------------------------- #
    # Introspection                                                      #
    # ---------------------------------------------------------------- #

    @property
    def is_ready(self) -> bool:
        """True after load() has completed successfully."""
        return True  # override if you track load state