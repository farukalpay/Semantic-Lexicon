# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Logging helpers for Semantic Lexicon."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(
    level: int = logging.INFO, logger_name: Optional[str] = None
) -> logging.Logger:
    """Configure application logging.

    Args:
        level: Logging level to apply.
        logger_name: Name of the logger to configure. Defaults to root.
    """

    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
