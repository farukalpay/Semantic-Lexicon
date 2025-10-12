"""Random seeding helpers."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def seed_everything(seed: Optional[int] = None) -> int:
    """Seed Python and NumPy RNGs."""

    if seed is None:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed
