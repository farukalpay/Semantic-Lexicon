"""Randomness helpers for deterministic behaviour."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

import numpy as np


def deterministic_hash(value: str) -> int:
    """Return a deterministic integer hash for ``value``."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def seed_everything(seed: Optional[int] = None) -> int:
    """Seed Python, NumPy, and OS randomness sources."""
    if seed is None:
        seed = deterministic_hash(os.getenv("SEMANTIC_LEXICON_SEED", "semantic-lexicon")) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    return seed
