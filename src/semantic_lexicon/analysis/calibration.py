# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Bayesian confidence calibration using a Dirichlet prior."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PosteriorPredictive:
    """Posterior parameters and predictive probabilities for intents."""

    alpha: np.ndarray
    predictive: np.ndarray


class DirichletCalibrator:
    """Perform Bayesian calibration with a symmetric or asymmetric Dirichlet prior.

    The calibrator maintains sufficient statistics ``n`` for observed intents and
    combines them with the prior parameters ``alpha``. The posterior predictive
    distribution equals ``(alpha + n) / (alpha.sum() + n.sum())`` and is the
    Bayes estimator under the absolute-error loss used by the expected
    calibration error metric.
    """

    def __init__(self, alpha: Iterable[float]):
        alpha = np.asarray(list(alpha), dtype=np.float64)
        if alpha.ndim != 1 or np.any(alpha <= 0):
            raise ValueError("alpha must be a positive 1-D vector")
        self._alpha = alpha
        self._counts = np.zeros_like(alpha)

    @property
    def num_intents(self) -> int:
        return int(self._alpha.size)

    @property
    def prior(self) -> np.ndarray:
        return self._alpha.copy()

    @property
    def counts(self) -> np.ndarray:
        return self._counts.copy()

    def update(self, intent: int) -> None:
        """Update the sufficient statistics with a newly observed intent label."""

        if intent < 0 or intent >= self.num_intents:
            raise ValueError("intent index out of bounds")
        self._counts[intent] += 1.0

    def batch_update(self, intents: Iterable[int]) -> None:
        for intent in intents:
            self.update(int(intent))

    def posterior(self) -> np.ndarray:
        """Return the posterior Dirichlet parameters ``alpha + counts``."""

        return self._alpha + self._counts

    def posterior_predictive(self) -> PosteriorPredictive:
        """Return the posterior parameters and predictive probabilities."""

        alpha_post = self.posterior()
        predictive = alpha_post / alpha_post.sum()
        return PosteriorPredictive(alpha=alpha_post, predictive=predictive)

    def calibrate(self, predicted: np.ndarray) -> np.ndarray:
        """Calibrate model probabilities via Bayesian convex combination.

        The calibrated distribution equals ``lambda * predicted + (1 - lambda) * m``
        where ``m`` is the posterior predictive mean and ``lambda`` shrinks towards
        one as the model confidence is supported by evidence. The shrinkage factor
        is ``lambda = (total_counts) / (total_counts + alpha.sum())`` which can be
        derived from minimising the expected calibration error with respect to the
        convex combination coefficient.
        """

        probs = np.asarray(predicted, dtype=np.float64)
        if probs.ndim != 1 or probs.size != self.num_intents:
            raise ValueError("predicted must be a 1-D probability vector")
        if np.any(probs < 0):
            raise ValueError("probabilities must be non-negative")
        total = self._counts.sum()
        shrinkage = total / (total + self._alpha.sum()) if total > 0 else 0.0
        posterior_mean = self.posterior_predictive().predictive
        mixture = shrinkage * probs + (1.0 - shrinkage) * posterior_mean
        mixture = np.clip(mixture, 0.0, None)
        if mixture.sum() == 0:
            mixture = posterior_mean
        else:
            mixture /= mixture.sum()
        return mixture
