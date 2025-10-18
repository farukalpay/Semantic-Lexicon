# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Semantic Lexicon package."""

from .algorithms import EXP3, AnytimeEXP3, EXP3Config
from .analysis import (
    DirichletCalibrator,
    PerformanceReport,
    PosteriorPredictive,
    PredictionSummary,
    RewardComponents,
    RobbinsMonroProcess,
    ValidationMetrics,
    ValidationRecord,
    benchmark_inference,
    composite_reward,
    composite_reward_bound,
    compute_confusion_correction,
    confidence_reward,
    confusion_correction_residual,
    convergence_rate_bound,
    correctness_reward,
    estimate_optimal_weights,
    evaluate_classifier,
    exp3_expected_regret,
    feedback_reward,
    load_validation_records,
    project_to_simplex,
    semantic_similarity_reward,
    simulate_intent_bandit,
    write_validation_report,
)
from .api import FeedbackAPI, FeedbackEvent, FeedbackService
from .config import SemanticModelConfig, load_config
from .go_no_go import (
    FairnessConfig,
    GoNoGoResult,
    KnowledgeCheckResult,
    KnowledgeSignals,
    PolicyLogEntry,
    SelectionSpec,
    StabilityCheckResult,
    run_go_no_go,
)
from .model import NeuralSemanticModel
from .presentation import BackupMove, ExperimentPlan, build_single_adjustment_plan
from .runtime import run
from .safety import (
    ConstraintSpec,
    GateBounds,
    GateTuningResult,
    ObjectiveSpec,
    SafetyGateTuner,
    run_primal_dual_autotune,
)
from .training import Trainer, TrainerConfig

__all__ = [
    "SemanticModelConfig",
    "TrainerConfig",
    "NeuralSemanticModel",
    "Trainer",
    "load_config",
    "EXP3",
    "EXP3Config",
    "AnytimeEXP3",
    "RewardComponents",
    "correctness_reward",
    "confidence_reward",
    "semantic_similarity_reward",
    "feedback_reward",
    "composite_reward",
    "composite_reward_bound",
    "estimate_optimal_weights",
    "benchmark_inference",
    "PerformanceReport",
    "DirichletCalibrator",
    "PosteriorPredictive",
    "exp3_expected_regret",
    "simulate_intent_bandit",
    "compute_confusion_correction",
    "confusion_correction_residual",
    "RobbinsMonroProcess",
    "convergence_rate_bound",
    "project_to_simplex",
    "ValidationRecord",
    "PredictionSummary",
    "ValidationMetrics",
    "load_validation_records",
    "evaluate_classifier",
    "write_validation_report",
    "FeedbackAPI",
    "FeedbackEvent",
    "FeedbackService",
    "run_go_no_go",
    "SelectionSpec",
    "PolicyLogEntry",
    "FairnessConfig",
    "GoNoGoResult",
    "KnowledgeSignals",
    "KnowledgeCheckResult",
    "StabilityCheckResult",
    "build_single_adjustment_plan",
    "ExperimentPlan",
    "BackupMove",
    "ObjectiveSpec",
    "ConstraintSpec",
    "GateBounds",
    "GateTuningResult",
    "SafetyGateTuner",
    "run_primal_dual_autotune",
    "run",
]

__version__ = "0.1.0"
