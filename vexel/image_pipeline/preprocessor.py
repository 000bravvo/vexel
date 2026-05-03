"""
Preprocessor — tiered blank-space detection and square-padding pipeline.

Architecture (§6.3 of the design doc):

    Input bytes / PIL Image
      → RGB conversion
      → Tier 1a: Alpha channel  (transparent → blank)
      → Tier 2 : Near-white / near-black fast-track
      → Tier 1 : Solid-colour corner-sampling
      → cv2.findContours → boundingRect
      → 5 % margin expansion (clamped to image bounds)
      → Crop to bounding box
      → Pad to square on neutral grey [128, 128, 128]
      → (optional) Resize to target_size
      → Return (PIL.Image, CropMetadata)

Edge-case guards:
  • All-blank image → raises BlankImageError (no vector generated)
  • Tiny subject (< min_subject_area fraction) → return full image,
    let CLIP CenterCrop handle it without extreme upscaling

CropMetadata is stored in the Qdrant payload so downstream consumers
can reconstruct where the product sits in the original image.
"""

from __future__ import annotations

import dataclasses
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # PIL.Image type hints are string-guarded below

from vexel.config import PreprocessorConfig


# ------------------------------------------------------------------ #
# Public data structures                                               #
# ------------------------------------------------------------------ #


@dataclasses.dataclass
class CropMetadata:
    """
    Bounding box of the detected content region on the original image.

    x, y, w, h are pixel coordinates on the *original* (pre-crop) image,
    making it possible for a frontend to highlight the matched region
    without reversing any transform.
    """

    x: int
    y: int
    w: int
    h: int
    processing_method: str
    """One of: "alpha" | "near_white" | "near_black" | "solid_color" | "none" | "disabled"."""

    was_cropped: bool = True
    """False when the full image was used (blank/tiny fallback)."""


class BlankImageError(ValueError):
    """Raised when smart_crop finds no detectable content in the image."""


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #


def _sample_corner_color(arr, corner_size: int = 5):
    """
    Return the mean RGB colour of the four corner regions as float32.

    Sampling a neighbourhood (not just one pixel) is more robust to
    JPEG/WebP compression artefacts at image edges.
    """
    import numpy as np

    h, w = arr.shape[:2]
    cs = max(1, min(corner_size, h // 4, w // 4))
    corners = [
        arr[:cs, :cs],
        arr[:cs, w - cs:],
        arr[h - cs:, :cs],
        arr[h - cs:, w - cs:],
    ]
    all_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return np.mean(all_pixels, axis=0).astype(np.float32)


# ------------------------------------------------------------------ #
# Preprocessor class                                                   #
# ------------------------------------------------------------------ #


class Preprocessor:
    """
    Blank-space removal + square-padding pipeline.

    Instantiate once per process and call process() per image.
    The pipeline is pure Python/NumPy/OpenCV — no GPU required.

    Example::

        preprocessor = Preprocessor(config.pipeline.preprocessor)
        pil_out, meta = preprocessor.process(image_bytes)
        # pil_out is a square RGB PIL.Image ready for encode()
    """

    def __init__(self, config: PreprocessorConfig) -> None:
        self._cfg = config

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def process(self, image_input) -> "tuple":
        """
        Detect blank space, crop to content, and pad to a grey square.

        Args:
            image_input: raw bytes *or* a PIL.Image.Image.

        Returns:
            (pil_image, CropMetadata)
            pil_image  — square RGB PIL Image, side = max(crop_w, crop_h).
            CropMetadata — bbox coords on the *original* image.

        Raises:
            BlankImageError: if no foreground content is detectable.
        """
        if not self._cfg.enabled:
            pil = self._to_pil(image_input)
            orig_w, orig_h = pil.size
            return pil, CropMetadata(0, 0, orig_w, orig_h, "disabled", was_cropped=False)

        pil = self._to_pil(image_input)
        return self._smart_crop(pil)

    # ---------------------------------------------------------------- #
    # Internal pipeline                                                  #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _to_pil(image_input):
        """Convert bytes or existing PIL Image to a PIL Image."""
        from PIL import Image

        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        return image_input

    def _smart_crop(self, pil_image) -> "tuple":
        """Full blank-space detection and crop pipeline."""
        import cv2
        import numpy as np
        from PIL import Image

        cfg = self._cfg
        pad_color = tuple(cfg.padding_color[:3])

        orig_w, orig_h = pil_image.size
        total_area = orig_w * orig_h

        # ── Tier 1a: Alpha channel ────────────────────────────────────
        has_alpha = pil_image.mode in ("RGBA", "LA") or (
            pil_image.mode == "P" and "transparency" in pil_image.info
        )
        img_rgb = pil_image.convert("RGB")
        arr = np.array(img_rgb)

        if has_alpha:
            alpha_arr = np.array(pil_image.convert("RGBA"))[:, :, 3]
            mask = ((alpha_arr > 10).astype(np.uint8)) * 255
            processing_method = "alpha"
        else:
            bg_color = _sample_corner_color(arr)
            r, g, b = float(bg_color[0]), float(bg_color[1]), float(bg_color[2])
            tol = cfg.near_neutral_tolerance

            # ── Tier 2: Near-white ────────────────────────────────────
            if all(v >= 255 - tol for v in (r, g, b)):
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 255 - tol, 255, cv2.THRESH_BINARY_INV)
                processing_method = "near_white"

            # ── Tier 2: Near-black ────────────────────────────────────
            elif all(v <= tol for v in (r, g, b)):
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, tol, 255, cv2.THRESH_BINARY)
                processing_method = "near_black"

            # ── Tier 1: Solid-colour BG via colour distance ───────────
            else:
                diff = np.abs(arr.astype(np.float32) - bg_color).sum(axis=2)
                mask = ((diff > cfg.bg_threshold).astype(np.uint8)) * 255
                processing_method = "solid_color"

        # ── Content bounding box ──────────────────────────────────────
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise BlankImageError(
                f"No foreground content detected via '{processing_method}' — "
                "image appears entirely blank."
            )

        all_pts = np.concatenate(contours, axis=0)
        x, y, w, h = cv2.boundingRect(all_pts)

        # ── Tiny-subject fallback ─────────────────────────────────────
        if w * h < total_area * cfg.min_subject_area:
            return img_rgb, CropMetadata(
                0, 0, orig_w, orig_h, "none", was_cropped=False
            )

        # ── Expand BBox by margin ─────────────────────────────────────
        mx = int(w * cfg.margin_pct)
        my = int(h * cfg.margin_pct)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(orig_w, x + w + mx)
        y2 = min(orig_h, y + h + my)

        # ── Crop ──────────────────────────────────────────────────────
        cropped = img_rgb.crop((x1, y1, x2, y2))
        cw, ch = cropped.size

        # ── Pad to square with neutral grey ───────────────────────────
        max_dim = max(cw, ch)
        square = Image.new("RGB", (max_dim, max_dim), pad_color)
        square.paste(cropped, ((max_dim - cw) // 2, (max_dim - ch) // 2))

        return square, CropMetadata(
            x=x1, y=y1, w=x2 - x1, h=y2 - y1,
            processing_method=processing_method,
            was_cropped=True,
        )