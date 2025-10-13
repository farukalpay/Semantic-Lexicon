"""Analytical helpers for reward shaping, calibration, and regret analysis."""

from .calibration import DirichletCalibrator, PosteriorPredictive
from .convergence import (
    RobbinsMonroProcess,
    convergence_rate_bound,
)
from .error import (
    compute_confusion_correction,
    confusion_correction_residual,
)
from .regret import (
    composite_reward_bound,
    exp3_expected_regret,
    simulate_intent_bandit,
)
from .reward import (
    RewardComponents,
    composite_reward,
    confidence_reward,
    correctness_reward,
    estimate_optimal_weights,
    feedback_reward,
    project_to_simplex,
    semantic_similarity_reward,
)

__all__ = [
    "RewardComponents",
    "composite_reward",
    "confidence_reward",
    "correctness_reward",
    "estimate_optimal_weights",
    "feedback_reward",
    "project_to_simplex",
    "semantic_similarity_reward",
    "DirichletCalibrator",
    "PosteriorPredictive",
    "composite_reward_bound",
    "exp3_expected_regret",
    "simulate_intent_bandit",
    "compute_confusion_correction",
    "confusion_correction_residual",
    "RobbinsMonroProcess",
    "convergence_rate_bound",
]
