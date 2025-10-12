"""Logging configuration for the Semantic Lexicon package."""

from __future__ import annotations

import logging
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, handler: Optional[logging.Handler] = None) -> None:
    """Configure root logging with a consistent format."""
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT, handlers=[handler] if handler else None)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the given ``name``."""
    return logging.getLogger(name)
