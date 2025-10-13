# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

import numpy as np
import pytest

from semantic_lexicon.analysis import (
    DirichletCalibrator,
    RewardComponents,
    RobbinsMonroProcess,
    composite_reward,
    composite_reward_bound,
    compute_confusion_correction,
    confusion_correction_residual,
    convergence_rate_bound,
    estimate_optimal_weights,
    exp3_expected_regret,
    simulate_intent_bandit,
)


@pytest.fixture
def historical_components() -> list[RewardComponents]:
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(20):
        correctness = rng.integers(0, 2)
        confidence = rng.uniform(0.4, 0.9)
        semantic = rng.uniform(0.5, 1.0)
        feedback = rng.uniform(0.0, 1.0)
        samples.append(
            RewardComponents(
                correctness=float(correctness),
                confidence=float(confidence),
                semantic=float(semantic),
                feedback=float(feedback),
            )
        )
    return samples


def test_reward_components_bounded(historical_components: list[RewardComponents]) -> None:
    weights = np.full(4, 0.25)
    for comp in historical_components:
        reward = composite_reward(comp, weights)
        assert 0.0 <= reward <= 1.0


def test_estimate_optimal_weights_simple_case() -> None:
    history = [
        RewardComponents(1.0, 0.8, 0.9, 0.4),
        RewardComponents(0.0, 0.3, 0.6, 0.7),
        RewardComponents(1.0, 0.7, 0.8, 0.6),
        RewardComponents(0.0, 0.4, 0.5, 0.2),
    ]
    realised = [0.82, 0.33, 0.76, 0.31]
    weights = estimate_optimal_weights(history, realised)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    reconstructed = [comp.as_array() @ weights for comp in history]
    assert np.allclose(reconstructed, realised, atol=0.06)


def test_dirichlet_calibration_reduces_ece() -> None:
    calibrator = DirichletCalibrator(alpha=[1.0, 1.0, 1.0, 1.0])
    calibrator.batch_update([0] * 12 + [1] * 8 + [2] * 5 + [3] * 5)
    posterior = calibrator.posterior_predictive()
    assert posterior.predictive.sum() == pytest.approx(1.0)
    raw_probs = np.array([0.9, 0.05, 0.03, 0.02])
    calibrated = calibrator.calibrate(raw_probs)
    assert calibrated.sum() == pytest.approx(1.0)
    accuracy = posterior.predictive[0]
    raw_ece = abs(accuracy - raw_probs[0])
    calibrated_ece = abs(accuracy - calibrated[0])
    assert calibrated_ece < raw_ece


def test_exp3_regret_simulation_bounded(historical_components: list[RewardComponents]) -> None:
    horizon = len(historical_components)
    num_arms = 4
    rng = np.random.default_rng(13)
    # Create per-arm variations by cyclically rotating the historical components.
    reward_sequences = []
    for t in range(horizon):
        base = historical_components[t]
        arms = [
            base,
            RewardComponents(
                base.correctness * 0.8,
                base.confidence,
                base.semantic,
                base.feedback,
            ),
            RewardComponents(
                base.correctness,
                base.confidence * 0.9,
                base.semantic,
                base.feedback,
            ),
            RewardComponents(
                base.correctness,
                base.confidence,
                base.semantic * 0.95,
                base.feedback,
            ),
        ]
        reward_sequences.append(arms)
    optimal_indices = [0] * horizon
    weights = np.array([0.4, 0.2, 0.2, 0.2])
    result = simulate_intent_bandit(reward_sequences, optimal_indices, weights, rng)
    assert result.regret[-1] <= exp3_expected_regret(num_arms, horizon)


def test_composite_reward_bound_checks_simplex() -> None:
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    assert composite_reward_bound(weights) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        composite_reward_bound(np.array([-0.1, 0.5, 0.3, 0.3]))
    with pytest.raises(ValueError):
        composite_reward_bound(np.array([0.5, 0.5, 0.5, -0.5]))
    with pytest.raises(ValueError):
        composite_reward_bound(np.array([0.5, 0.5, 0.5, 0.1]))


def test_confusion_correction_reduces_residual() -> None:
    confusion = np.array(
        [
            [30, 5, 3, 2],
            [6, 22, 7, 5],
            [4, 6, 18, 8],
            [3, 5, 7, 25],
        ],
        dtype=float,
    )
    identity_residual = confusion_correction_residual(confusion, np.eye(4))
    transform = compute_confusion_correction(confusion)
    corrected_residual = confusion_correction_residual(confusion, transform)
    assert corrected_residual < identity_residual


def test_robbins_monro_convergence_rate() -> None:
    process = RobbinsMonroProcess(
        step_schedule=lambda t: 1.0 / (t + 5.0),
        operator=lambda theta, noise: -theta + noise,
    )
    theta0 = np.array([1.0, -0.5])
    trajectory = process.iterate(theta0, [np.zeros_like(theta0) for _ in range(50)])
    assert np.linalg.norm(trajectory[-1]) < np.linalg.norm(theta0)
    bound = convergence_rate_bound(lipschitz=0.5, variance=0.1, horizon=50)
    assert bound > 0.0
