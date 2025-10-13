# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Reward engineering helpers with theoretical guarantees."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardComponents:
    """Container for the four reward components used by the bandit loop."""

    correctness: float
    confidence: float
    semantic: float
    feedback: float

    def as_array(self) -> np.ndarray:
        """Return the components as a NumPy array.

        All components are clipped into ``[0, 1]`` so downstream code can rely on
        boundedness when combining them.
        """

        values = np.array(
            [self.correctness, self.confidence, self.semantic, self.feedback],
            dtype=np.float64,
        )
        np.clip(values, 0.0, 1.0, out=values)
        return values


def correctness_reward(selected: int, optimal: int) -> float:
    """Return the correctness indicator ``1{selected == optimal}``."""

    return float(selected == optimal)


def confidence_reward(prob_selected: float, selected: int, optimal: int) -> float:
    """Return a calibration-aware confidence reward.

    The function realises ``1 - |p - 1{selected == optimal}|``, which equals the
    probability assigned to the realised outcome in the well-calibrated case and
    penalises over-confident mistakes. The value is always in ``[0, 1]``.
    """

    indicator = 1.0 if selected == optimal else 0.0
    return 1.0 - abs(float(prob_selected) - indicator)


def semantic_similarity_reward(similarity: float) -> float:
    """Return a semantic similarity score clipped into ``[0, 1]``."""

    return float(np.clip(similarity, 0.0, 1.0))


def feedback_reward(feedback: float) -> float:
    """Return user feedback clipped into ``[0, 1]``."""

    return float(np.clip(feedback, 0.0, 1.0))


def composite_reward(components: RewardComponents, weights: Sequence[float]) -> float:
    """Compute ``R = w^T r`` with ``w`` on the probability simplex.

    Parameters
    ----------
    components:
        The individual reward components, each in ``[0, 1]``.
    weights:
        Non-negative weights that must sum to one. When the weights do not
        exactly satisfy the constraints because of numerical noise, they are
        projected onto the simplex before being applied.
    """

    raw_weights = np.asarray(weights, dtype=np.float64)
    if raw_weights.ndim != 1 or raw_weights.size != 4:
        raise ValueError("Expected four weights matching the reward components")
    projected = project_to_simplex(raw_weights)
    return float(projected @ components.as_array())


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project ``vector`` onto the probability simplex using the method of [1].

    References
    ----------
    [1] Wang, Weiran, and Miguel Á. Carreira-Perpiñán. "Projection onto the
        probability simplex: An efficient algorithm with a simple proof, and an
        application." arXiv preprint arXiv:1309.1541 (2013).
    """

    if vector.ndim != 1:
        raise ValueError("Simplex projection expects a 1-D vector")
    if np.all(vector == 0):
        return np.full_like(vector, 1.0 / vector.size)

    u = np.sort(vector)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, vector.size + 1) > (cssv - 1))[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(vector - theta, 0.0)
    # Ensure exact normalisation.
    if not np.isclose(w.sum(), 1.0):
        w /= w.sum()
    return w


def estimate_optimal_weights(
    history: Iterable[RewardComponents],
    realised_rewards: Sequence[float],
) -> np.ndarray:
    r"""Estimate optimal weights by constrained least squares.

    Given a historical corpus of component vectors ``r_t`` and realised rewards
    ``y_t``, solve

    .. math::

       \min_{w \in \Delta^3} \sum_t (w^\top r_t - y_t)^2,

    where ``\Delta^3`` denotes the probability simplex. The solution is computed
    by projecting the ordinary least-squares solution onto the simplex, which is
    the unique optimal point for this strictly convex programme.
    """

    r_matrix = np.vstack([comp.as_array() for comp in history])
    y = np.asarray(realised_rewards, dtype=np.float64)
    if r_matrix.shape[0] != y.size:
        raise ValueError("history and realised_rewards must share the same length")
    # Regularised least squares for numerical stability.
    reg = 1e-6
    gram = r_matrix.T @ r_matrix + reg * np.eye(r_matrix.shape[1])
    rhs = r_matrix.T @ y
    solution = np.linalg.solve(gram, rhs)
    return project_to_simplex(solution)
