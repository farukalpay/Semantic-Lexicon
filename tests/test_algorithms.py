# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

import numpy as np
import pytest

from semantic_lexicon.algorithms import EXP3, AnytimeEXP3, EXP3Config


def test_exp3_initialises_uniform_probabilities() -> None:
    config = EXP3Config(num_arms=2, horizon=8, rng=np.random.default_rng(0))
    solver = EXP3(config)
    assert np.allclose(solver.probabilities, np.full(2, 0.5))


def test_exp3_weight_update_boosts_selected_arm() -> None:
    rng = np.random.default_rng(1)
    config = EXP3Config(num_arms=3, horizon=5, rng=rng)
    solver = EXP3(config)
    weights_before = solver.weights
    arm = solver.select_arm()
    solver.update(1.0)
    weights_after = solver.weights
    assert weights_after[arm] > weights_before[arm]
    untouched_indices = [i for i in range(3) if i != arm]
    assert np.allclose(weights_after[untouched_indices], weights_before[untouched_indices])


def test_exp3_rejects_rewards_outside_unit_interval() -> None:
    config = EXP3Config(num_arms=2, horizon=3, rng=np.random.default_rng(2))
    solver = EXP3(config)
    solver.select_arm()
    with pytest.raises(ValueError):
        solver.update(1.5)


def test_exp3_enforces_horizon_limit() -> None:
    config = EXP3Config(num_arms=2, horizon=1, rng=np.random.default_rng(3))
    solver = EXP3(config)
    solver.select_arm()
    solver.update(0.5)
    with pytest.raises(RuntimeError):
        solver.select_arm()


def test_anytime_exp3_doubles_epoch_horizon() -> None:
    rng = np.random.default_rng(4)
    agent = AnytimeEXP3(num_arms=2, rng=rng)
    horizons = []
    for _ in range(4):
        horizons.append(agent.epoch_horizon)
        agent.select_arm()
        agent.update(0.0)
    assert horizons == [1, 2, 2, 4]


def test_anytime_exp3_resets_distribution_between_epochs() -> None:
    rng = np.random.default_rng(5)
    agent = AnytimeEXP3(num_arms=2, rng=rng)
    agent.select_arm()
    agent.update(1.0)
    assert np.allclose(agent.probabilities, np.full(2, 0.5))
