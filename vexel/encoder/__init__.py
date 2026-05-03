"""vexel.encoder — encoder abstraction and implementations."""

from vexel.encoder.base import BaseEncoder
from vexel.encoder.factory import get_encoder

__all__ = ["BaseEncoder", "get_encoder"]