"""
Encoder factory — instantiate the right encoder from config.

Add support for new encoders here without touching any other module.
"""

from __future__ import annotations

from vexel.config import EncoderConfig
from vexel.encoder.base import BaseEncoder


def get_encoder(config: EncoderConfig) -> BaseEncoder:
    """
    Return an encoder instance for the given config.

    Args:
        config: EncoderConfig with .name, .model, .device, etc.

    Returns:
        Unloaded BaseEncoder subclass instance.  Call await encoder.load()
        before use.

    Raises:
        ValueError: for unknown encoder names.
    """
    name = config.name.lower()

    if name == "clip":
        from vexel.encoder.clip import CLIPEncoder
        return CLIPEncoder(config)

    if name == "siglip":
        raise NotImplementedError(
            "SigLIPEncoder is on the Phase 2 roadmap.  "
            "Implement BaseEncoder and register it here."
        )

    raise ValueError(
        f"Unknown encoder '{config.name}'.  "
        f"Supported values: 'clip'.  "
        f"For custom encoders, subclass BaseEncoder and pass the instance "
        f"directly to MultiVectorIndexer / VisualSearchEngine."
    )