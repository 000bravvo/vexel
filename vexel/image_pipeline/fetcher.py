"""
ImageFetcher — HTTP image download with CDN URL transformation and retry.

The CDN transform is driven by the template string in PipelineConfig so
that the same library code works across Gumlet, Cloudinary, Imgix, or
any other CDN that supports resize-on-the-fly query parameters.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Callable

from vexel.config import CDNTransformConfig, PreprocessorConfig


def build_cdn_url(
    raw_url: str,
    cdn_config: CDNTransformConfig,
    size: int,
) -> str:
    """
    Rewrite a raw image URL using the configured CDN template.

    The template supports two placeholders:
        {url}   — the raw image URL (verbatim)
        {size}  — the integer pixel dimension (same value for w and h)

    Examples:
        Gumlet:     "{url}?w={size}&h={size}"      → url?w=512&h=512
        Cloudinary: "{url}/c_fit,w_{size},h_{size}" → url/c_fit,w_512,h_512
        Imgix:      "{url}?w={size}&h={size}&fit=clip"

    Args:
        raw_url:    original CDN URL, may already contain query params.
        cdn_config: CDNTransformConfig from VexelConfig.
        size:       target pixel dimension.

    Returns:
        Rewritten URL string.  Returns raw_url unchanged if cdn is disabled.
    """
    if not cdn_config.enabled or not raw_url:
        return raw_url
    return cdn_config.template.format(url=raw_url, size=size)


class ImageFetcher:
    """
    Synchronous image downloader with retry logic.

    Intentionally synchronous so it can be run inside a ThreadPoolExecutor
    alongside the encoder — both are I/O / CPU bound and should not block
    the asyncio event loop.
    """

    def __init__(
        self,
        cdn_config: CDNTransformConfig,
        preprocessor_config: PreprocessorConfig,
        timeout: int = 10,
        retries: int = 2,
        url_transform_fn: Callable[[str], str] | None = None,
    ) -> None:
        """
        Args:
            cdn_config:         controls URL template rewriting.
            preprocessor_config: needed for the download_size parameter.
            timeout:            per-request timeout in seconds.
            retries:            number of retry attempts after failure.
            url_transform_fn:   optional custom transform applied AFTER the CDN
                                template, e.g. to sign URLs or add auth tokens.
        """
        self._cdn_config = cdn_config
        self._download_size = preprocessor_config.download_size
        self._timeout = timeout
        self._retries = retries
        self._url_transform_fn = url_transform_fn

    def get_download_url(self, raw_url: str) -> str:
        """
        Apply the CDN transform (and optional custom transform) to raw_url.

        Returns the URL that will actually be fetched.
        """
        if not raw_url:
            return raw_url
        url = build_cdn_url(raw_url, self._cdn_config, self._download_size)
        if self._url_transform_fn:
            url = self._url_transform_fn(url)
        return url

    def fetch(self, raw_url: str) -> bytes:
        """
        Download image bytes for the given raw URL.

        Applies CDN transform first, then fetches with exponential back-off.

        Args:
            raw_url: the original (unmodified) image URL.

        Returns:
            Raw image bytes.

        Raises:
            urllib.error.HTTPError: on non-2xx response after all retries.
            TimeoutError:          if every attempt times out.
        """
        url = self.get_download_url(raw_url)
        last_exc: Exception = RuntimeError("no attempts made")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "vexel-image-fetcher/0.1"},
        )

        for attempt in range(self._retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return resp.read()
            except Exception as exc:
                last_exc = exc
                if attempt < self._retries:
                    time.sleep(0.5 * (2**attempt))  # 0.5 s, 1 s back-off

        raise last_exc