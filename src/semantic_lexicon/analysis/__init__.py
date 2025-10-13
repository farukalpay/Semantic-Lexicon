# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

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
from .performance import PerformanceReport, benchmark_inference
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
from .validation import (
    PredictionSummary,
    ValidationMetrics,
    ValidationRecord,
    evaluate_classifier,
    load_validation_records,
    write_validation_report,
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
    "PerformanceReport",
    "benchmark_inference",
    "DirichletCalibrator",
    "PosteriorPredictive",
    "composite_reward_bound",
    "exp3_expected_regret",
    "simulate_intent_bandit",
    "compute_confusion_correction",
    "confusion_correction_residual",
    "RobbinsMonroProcess",
    "convergence_rate_bound",
    "ValidationRecord",
    "PredictionSummary",
    "ValidationMetrics",
    "load_validation_records",
    "evaluate_classifier",
    "write_validation_report",
]
