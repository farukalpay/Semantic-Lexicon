# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Systematic error analysis utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_confusion_correction(
    confusion: np.ndarray, damping: float = 1e-6
) -> NDArray[np.float64]:
    """Return the transformation matrix that minimises ``||C T - I||_F``.

    The solution is the Moore-Penrose pseudoinverse ``C^+``; we stabilise the
    inversion by applying ``damping`` to singular values that are nearly zero.
    """

    c = np.asarray(confusion, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("confusion must be a square matrix")
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    s_inv = 1.0 / (s + damping)
    t = vt.T @ np.diag(s_inv) @ u.T
    return np.asarray(t, dtype=np.float64)


def confusion_correction_residual(confusion: np.ndarray, transform: np.ndarray) -> float:
    """Return the Frobenius norm of ``C T - I`` to quantify residual error."""

    c = np.asarray(confusion, dtype=np.float64)
    t = np.asarray(transform, dtype=np.float64)
    identity = np.eye(c.shape[0])
    residual = c @ t - identity
    return float(np.linalg.norm(residual, ord="fro"))
