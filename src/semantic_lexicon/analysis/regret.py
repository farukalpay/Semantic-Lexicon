# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Regret analysis utilities for EXP3 in the intent-routing setting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from semantic_lexicon.algorithms import EXP3, EXP3Config
from semantic_lexicon.analysis.reward import RewardComponents, composite_reward


@dataclass(frozen=True)
class RegretResult:
    """Holds the cumulative regret trajectory of a simulation."""

    rewards: np.ndarray
    optimal: np.ndarray
    regret: np.ndarray


def composite_reward_bound(weights: Sequence[float]) -> float:
    """Return the tight upper bound on the composite reward.

    Because every component lies in ``[0, 1]`` and the weights are on the simplex,
    the composite reward is always contained in ``[0, 1]``. The function returns
    the numerical bound ``1.0`` to make the dependency explicit in code and tests.
    """

    projected = np.asarray(weights, dtype=np.float64)
    if projected.ndim != 1 or projected.size != 4:
        raise ValueError("Expected four weights")
    if np.any(projected < 0):
        raise ValueError("Weights must be non-negative")
    total = projected.sum()
    if not np.isclose(total, 1.0):
        raise ValueError("Weights must sum to one to stay on the simplex")
    return 1.0


def exp3_expected_regret(num_arms: int, horizon: int) -> float:
    """Return the standard regret upper bound for EXP3."""

    if num_arms < 2 or horizon <= 0:
        raise ValueError("num_arms must be >=2 and horizon > 0")
    return 2.63 * math.sqrt(num_arms * horizon * math.log(num_arms))


def simulate_intent_bandit(
    reward_sequences: Sequence[Sequence[RewardComponents]],
    optimal_indices: Sequence[int],
    weights: Sequence[float],
    rng: np.random.Generator | None = None,
) -> RegretResult:
    """Simulate EXP3 on deterministic reward sequences for regret estimation.

    Parameters
    ----------
    reward_sequences:
        ``reward_sequences[t][i]`` contains the reward components for arm ``i`` at
        round ``t``.
    optimal_indices:
        Indices of the optimal arm at each round.
    weights:
        Composite reward weights used to convert components into scalar rewards.
    rng:
        Optional ``Generator`` passed to the EXP3 implementation for repeatable
        sampling.
    """

    horizon = len(reward_sequences)
    if horizon == 0:
        raise ValueError("At least one round is required for the simulation")
    num_arms = len(reward_sequences[0])
    if num_arms < 2:
        raise ValueError("Need at least two arms to run the bandit simulation")
    if len(optimal_indices) != horizon:
        raise ValueError("optimal_indices length must match the horizon")

    solver = EXP3(EXP3Config(num_arms=num_arms, horizon=horizon, rng=rng))
    realised = np.zeros(horizon, dtype=np.float64)
    optimal = np.zeros(horizon, dtype=np.float64)

    for t in range(horizon):
        arm = solver.select_arm()
        arm_reward = composite_reward(reward_sequences[t][arm], weights)
        solver.update(arm_reward)
        realised[t] = arm_reward
        optimal[t] = composite_reward(reward_sequences[t][optimal_indices[t]], weights)

    regret = optimal.cumsum() - realised.cumsum()
    return RegretResult(rewards=realised, optimal=optimal, regret=regret)
