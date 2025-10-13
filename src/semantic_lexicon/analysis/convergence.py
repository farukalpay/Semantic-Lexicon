"""Convergence analysis for the coupled classifier and bandit system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class RobbinsMonroProcess:
    """Representation of ``theta_{t+1} = theta_t + gamma_t H(theta_t, X_t)``."""

    step_schedule: Callable[[int], float]
    operator: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def iterate(self, theta0: np.ndarray, noise_sequence: list[np.ndarray]) -> list[np.ndarray]:
        """Run the stochastic approximation for a prescribed noise sequence."""

        theta = np.asarray(theta0, dtype=np.float64)
        trajectory = [theta.copy()]
        for t, noise in enumerate(noise_sequence, start=1):
            gamma_t = self.step_schedule(t)
            theta = theta + gamma_t * self.operator(theta, noise)
            trajectory.append(theta.copy())
        return trajectory


def convergence_rate_bound(lipschitz: float, variance: float, horizon: int) -> float:
    """Return a high-level ``O(1 / sqrt{n})`` bound on the estimation error."""

    if lipschitz <= 0 or variance < 0 or horizon <= 0:
        raise ValueError("Inputs must be positive and horizon must be > 0")
    return np.sqrt(variance / (lipschitz * horizon))
