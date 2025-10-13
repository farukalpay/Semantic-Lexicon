# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""EXP3 adversarial bandit algorithms.

This module implements the EXP3 algorithm for adversarial bandits with
bounded rewards as well as an anytime variant obtained through the
classic doubling trick.  The implementation follows the pseudocode in
literature (e.g. Auer et al., 2002) and is tailored to the style
selection setting described in the project documentation where two
personality "styles" compete, but the code supports an arbitrary number
of arms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EXP3Config:
    r"""Configuration parameters for :class:`EXP3`.

    Parameters
    ----------
    num_arms:
        Number of available actions ("styles"). Must be at least two.
    horizon:
        Number of rounds to run in the current epoch. Required because
        the learning rate depends on it.
    eta:
        Optional override for the learning rate. When omitted the
        canonical choice :math:`\sqrt{2 \ln K / (K T)}` is used.
    gamma:
        Optional override for the exploration mass. By default the value
        is ``min(1, K * eta)`` which guarantees sufficient exploration.
    rng:
        Optional ``numpy.random.Generator`` used for arm selection. When
        omitted a fresh generator is created via ``default_rng``.
    """

    num_arms: int = 2
    horizon: int = 1
    eta: Optional[float] = None
    gamma: Optional[float] = None
    rng: Optional[np.random.Generator] = None


class EXP3:
    """Adversarial bandit algorithm with importance-weighted updates.

    The implementation keeps the textbook semantics: after calling
    :meth:`select_arm` to obtain the next action, supply the observed
    reward via :meth:`update`. Rewards are expected to lie in ``[0, 1]``
    and will raise an error otherwise. The internal weights are
    initialised to one which yields a uniform distribution before the
    first update.
    """

    def __init__(self, config: EXP3Config) -> None:
        if config.num_arms < 2:
            raise ValueError("EXP3 requires at least two arms.")
        if config.horizon <= 0:
            raise ValueError("The horizon must be a positive integer.")

        self._num_arms = config.num_arms
        self._horizon = config.horizon
        self._eta = (
            config.eta
            if config.eta is not None
            else math.sqrt((2.0 * math.log(self._num_arms)) / (self._num_arms * self._horizon))
        )
        if self._eta <= 0:
            raise ValueError("The learning rate eta must be positive.")
        self._gamma = (
            config.gamma if config.gamma is not None else min(1.0, self._num_arms * self._eta)
        )
        if not 0 < self._gamma <= 1:
            raise ValueError("gamma must lie in (0, 1].")

        self._rng = config.rng or np.random.default_rng()
        self._weights = np.ones(self._num_arms, dtype=np.float64)
        self._probabilities = self._compute_probabilities()
        self._round = 0
        self._last_arm: Optional[int] = None

    @property
    def horizon(self) -> int:
        """Number of rounds allocated to this EXP3 instance."""

        return self._horizon

    @property
    def num_arms(self) -> int:
        """Return the number of available actions."""

        return self._num_arms

    @property
    def eta(self) -> float:
        """Return the current learning rate."""

        return self._eta

    @property
    def gamma(self) -> float:
        """Return the current exploration mass."""

        return self._gamma

    @property
    def weights(self) -> np.ndarray:
        """Return a copy of the internal weight vector."""

        return self._weights.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Return the sampling distribution used to draw the next arm."""

        return self._probabilities.copy()

    def _compute_probabilities(self) -> np.ndarray:
        normaliser = float(self._weights.sum())
        if normaliser <= 0:
            raise RuntimeError("Internal weights became non-positive, cannot normalise.")
        base_distribution = self._weights / normaliser
        exploration = np.full(self._num_arms, 1.0 / self._num_arms, dtype=np.float64)
        probabilities = (1.0 - self._gamma) * base_distribution + self._gamma * exploration
        return probabilities

    def select_arm(self) -> int:
        """Sample an arm according to the current probabilities."""

        if self._round >= self._horizon:
            raise RuntimeError("The horizon for this EXP3 instance has been exhausted.")

        arm = int(self._rng.choice(self._num_arms, p=self._probabilities))
        self._last_arm = arm
        return arm

    def update(self, reward: float) -> None:
        """Update the weight vector using the observed reward.

        Parameters
        ----------
        reward:
            Realised reward for the previously selected arm. Must lie in
            ``[0, 1]`` as assumed by the regret analysis of EXP3.
        """

        if self._last_arm is None:
            raise RuntimeError("select_arm must be called before update.")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("EXP3 expects rewards in the interval [0, 1].")

        estimates = np.zeros(self._num_arms, dtype=np.float64)
        estimates[self._last_arm] = reward / self._probabilities[self._last_arm]
        self._weights *= np.exp(self._eta * estimates)
        self._probabilities = self._compute_probabilities()
        self._round += 1
        self._last_arm = None


class AnytimeEXP3:
    """Anytime variant of EXP3 realised through the doubling trick.

    The instance runs a sequence of epochs. Each epoch instantiates a
    fresh :class:`EXP3` solver with a doubled horizon. Users interact via
    the same :meth:`select_arm` and :meth:`update` methods; the class
    handles epoch transitions transparently.
    """

    def __init__(self, num_arms: int = 2, rng: Optional[np.random.Generator] = None) -> None:
        if num_arms < 2:
            raise ValueError("AnytimeEXP3 requires at least two arms.")

        self._num_arms = num_arms
        self._rng = rng or np.random.default_rng()
        self._epoch_index = 0
        self._epoch_horizon = 1
        self._steps_in_epoch = 0
        self._solver = self._spawn_solver()

    @property
    def epoch(self) -> int:
        """Return the current epoch index (starting from zero)."""

        return self._epoch_index

    @property
    def epoch_horizon(self) -> int:
        """Number of rounds allocated to the active epoch."""

        return self._epoch_horizon

    @property
    def probabilities(self) -> np.ndarray:
        """Return the sampling distribution of the active EXP3 solver."""

        return self._solver.probabilities

    def _spawn_solver(self) -> EXP3:
        config = EXP3Config(
            num_arms=self._num_arms,
            horizon=self._epoch_horizon,
            rng=self._rng,
        )
        return EXP3(config)

    def select_arm(self) -> int:
        """Delegate to the active EXP3 solver."""

        return self._solver.select_arm()

    def update(self, reward: float) -> None:
        """Propagate the observed reward to the current solver and step epochs."""

        self._solver.update(reward)
        self._steps_in_epoch += 1
        if self._steps_in_epoch >= self._epoch_horizon:
            self._epoch_index += 1
            self._epoch_horizon *= 2
            self._steps_in_epoch = 0
            self._solver = self._spawn_solver()
