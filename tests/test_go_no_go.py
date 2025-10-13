"""Tests for the Go/No-Go evaluation suite."""

from __future__ import annotations

import math

import numpy as np

from semantic_lexicon.go_no_go import (
    FairnessConfig,
    KnowledgeSignals,
    PolicyLogEntry,
    SelectionSpec,
    run_go_no_go,
)


def _policy_probability(
    scores: np.ndarray,
    penalty: np.ndarray,
    prior: np.ndarray,
    eta: float,
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    logits = (scores - penalty + eta * prior) / temperature
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    base = exps / np.sum(exps)
    return (1.0 - epsilon) * base + epsilon / scores.size


def test_go_no_go_accepts_when_all_checks_pass() -> None:
    selection = SelectionSpec(
        selected=("ai coach", "speech timing", "feedback loop"),
        group_membership={
            "ai coach": ("off-topic", "R"),
            "speech timing": ("on-topic", "P"),
            "feedback loop": ("on-topic", "E"),
        },
        group_bounds={
            "on-topic": (2, 3),
            "off-topic": (1, 1),
        },
        knowledge=KnowledgeSignals(
            coverage=0.92,
            cohesion=0.81,
            knowledge_raw=0.88,
            knowledge_calibrated=0.76,
            baseline_coverage=0.9,
            baseline_cohesion=0.8,
            baseline_knowledge_calibrated_median=0.7,
        ),
    )
    scores = np.array([1.4, 0.6, 0.2])
    penalty = np.array([0.2, 0.1, 0.05])
    prior = np.array([0.3, 0.1, -0.05])
    eta = 0.5
    temperature = 0.9
    epsilon = 0.1
    policy = _policy_probability(scores, penalty, prior, eta, temperature, epsilon)
    logs = [
        PolicyLogEntry(
            agent="alpha",
            scores=scores,
            temperature=temperature,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=eta,
            penalty=penalty,
            penalty_mode="price",
            selected_action=0,
            logged_probability=float(policy[0]),
            reward=0.8,
            weight_floor=epsilon / scores.size,
            metrics=(0.52, 0.48),
            timestep=0,
        ),
        PolicyLogEntry(
            agent="beta",
            scores=scores,
            temperature=temperature,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=eta,
            penalty=penalty,
            penalty_mode="price",
            selected_action=1,
            logged_probability=float(policy[1]),
            reward=0.7,
            weight_floor=epsilon / scores.size,
            metrics=(0.5, 0.5),
            timestep=1,
        ),
    ]
    fairness = FairnessConfig(
        action_target=(0.5, 0.3, 0.2),
        action_tolerance=0.7,
        metric_target=(0.5, 0.5),
        metric_tolerance=0.1,
    )
    result = run_go_no_go(
        selection,
        logs,
        fairness=fairness,
        baseline_value=0.6,
        ess_min_ratio=0.01,
        stability_delta=0.1,
        stability_window=5,
    )
    assert result.accepted
    assert math.isclose(result.off_policy.snips, result.off_policy.snips)
    assert result.policy.mode == "price"
    assert result.off_policy.fairness_ok
    assert all(gap == 0.0 for gap in result.off_policy.fairness_gaps.values())
    assert result.knowledge.ok
    assert result.stability.ok


def test_go_no_go_rejects_on_rule_or_policy_failure() -> None:
    selection = SelectionSpec(
        selected=("ai coach", "speech timing"),
        group_membership={
            "ai coach": ("off-topic",),
            "speech timing": ("on-topic",),
        },
        group_bounds={
            "on-topic": (2, 3),
            "off-topic": (0, 0),
        },
        knowledge=KnowledgeSignals(
            coverage=0.75,
            cohesion=0.62,
            knowledge_raw=0.7,
            knowledge_calibrated=0.55,
            baseline_coverage=0.7,
            baseline_cohesion=0.6,
            baseline_knowledge_calibrated_median=0.5,
        ),
    )
    scores = np.array([0.8, 0.4])
    prior = np.zeros_like(scores)
    epsilon = 0.2
    logs = [
        PolicyLogEntry(
            agent="alpha",
            scores=scores,
            temperature=1.0,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=0.0,
            penalty=np.zeros_like(scores),
            penalty_mode="price",
            selected_action=0,
            logged_probability=0.6,
            reward=0.2,
            timestep=0,
        ),
        PolicyLogEntry(
            agent="alpha",
            scores=scores,
            temperature=1.0,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=0.0,
            penalty=np.zeros_like(scores),
            penalty_mode="congestion",
            selected_action=1,
            logged_probability=0.4,
            reward=0.1,
            timestep=1,
        ),
    ]
    result = run_go_no_go(selection, logs, baseline_value=0.5)
    assert not result.selection.feasible
    assert not result.policy.ok
    assert not result.accepted


def test_go_no_go_rejects_on_knowledge_lift_drop() -> None:
    selection = SelectionSpec(
        selected=("ai coach", "speech timing", "feedback loop"),
        group_membership={
            "ai coach": ("off-topic", "R"),
            "speech timing": ("on-topic", "P"),
            "feedback loop": ("on-topic", "E"),
        },
        group_bounds={
            "on-topic": (2, 3),
            "off-topic": (1, 1),
        },
        knowledge=KnowledgeSignals(
            coverage=0.88,
            cohesion=0.74,
            knowledge_raw=0.8,
            knowledge_calibrated=0.6,
            baseline_coverage=0.9,
            baseline_cohesion=0.76,
            baseline_knowledge_calibrated_median=0.65,
        ),
    )
    scores = np.array([1.0, 0.5, 0.3])
    penalty = np.array([0.1, 0.05, 0.02])
    prior = np.array([0.2, 0.0, -0.05])
    eta = 0.4
    temperature = 1.0
    epsilon = 0.15
    policy = _policy_probability(scores, penalty, prior, eta, temperature, epsilon)
    logs = [
        PolicyLogEntry(
            agent="alpha",
            scores=scores,
            temperature=temperature,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=eta,
            penalty=penalty,
            penalty_mode="price",
            selected_action=0,
            logged_probability=float(policy[0]),
            reward=0.6,
            weight_floor=epsilon / scores.size,
            timestep=0,
        ),
        PolicyLogEntry(
            agent="beta",
            scores=scores,
            temperature=temperature,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=eta,
            penalty=penalty,
            penalty_mode="price",
            selected_action=1,
            logged_probability=float(policy[1]),
            reward=0.55,
            weight_floor=epsilon / scores.size,
            timestep=1,
        ),
    ]
    result = run_go_no_go(
        selection,
        logs,
        baseline_value=0.5,
        ess_min_ratio=0.01,
        stability_delta=0.1,
        stability_window=5,
    )
    assert result.selection.feasible
    assert result.policy.ok
    assert result.off_policy.ok
    assert not result.knowledge.ok
    assert not result.accepted


def test_go_no_go_rejects_on_invalid_knowledge_weight() -> None:
    selection = SelectionSpec(
        selected=("ai coach", "speech timing", "feedback loop"),
        group_membership={
            "ai coach": ("off-topic", "R"),
            "speech timing": ("on-topic", "P"),
            "feedback loop": ("on-topic", "E"),
        },
        group_bounds={
            "on-topic": (2, 3),
            "off-topic": (1, 1),
        },
        knowledge=KnowledgeSignals(
            coverage=0.92,
            cohesion=0.81,
            knowledge_raw=0.88,
            knowledge_calibrated=0.76,
            baseline_coverage=0.9,
            baseline_cohesion=0.8,
            baseline_knowledge_calibrated_median=0.7,
        ),
    )
    scores = np.array([1.2, 0.7, 0.4])
    penalty = np.array([0.1, 0.05, 0.02])
    prior = np.array([0.2, 0.0, -0.05])
    eta = 1.2
    temperature = 0.9
    epsilon = 0.1
    policy = _policy_probability(scores, penalty, prior, eta, temperature, epsilon)
    logs = [
        PolicyLogEntry(
            agent="alpha",
            scores=scores,
            temperature=temperature,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=eta,
            penalty=penalty,
            penalty_mode="price",
            selected_action=0,
            logged_probability=float(policy[0]),
            reward=0.8,
            weight_floor=epsilon / scores.size,
            timestep=0,
        ),
    ]
    result = run_go_no_go(
        selection,
        logs,
        baseline_value=0.6,
        ess_min_ratio=0.01,
        stability_delta=0.1,
        stability_window=5,
    )
    assert not result.policy.ok
    assert not result.accepted


def test_go_no_go_rejects_when_weight_floor_below_exploration() -> None:
    selection = SelectionSpec(
        selected=("ai coach", "speech timing", "feedback loop"),
        group_membership={
            "ai coach": ("off-topic", "R"),
            "speech timing": ("on-topic", "P"),
            "feedback loop": ("on-topic", "E"),
        },
        group_bounds={
            "on-topic": (2, 3),
            "off-topic": (1, 1),
        },
        knowledge=KnowledgeSignals(
            coverage=0.92,
            cohesion=0.81,
            knowledge_raw=0.88,
            knowledge_calibrated=0.76,
            baseline_coverage=0.9,
            baseline_cohesion=0.8,
            baseline_knowledge_calibrated_median=0.7,
        ),
    )
    scores = np.array([1.0, 0.5, 0.3])
    penalty = np.array([0.1, 0.05, 0.02])
    prior = np.array([0.2, 0.0, -0.05])
    eta = 0.4
    temperature = 1.0
    epsilon = 0.15
    policy = _policy_probability(scores, penalty, prior, eta, temperature, epsilon)
    below_floor = (epsilon / scores.size) / 2.0
    logs = [
        PolicyLogEntry(
            agent="alpha",
            scores=scores,
            temperature=temperature,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=eta,
            penalty=penalty,
            penalty_mode="price",
            selected_action=0,
            logged_probability=float(policy[0]),
            reward=0.8,
            weight_floor=below_floor,
            timestep=0,
        ),
        PolicyLogEntry(
            agent="beta",
            scores=scores,
            temperature=temperature,
            epsilon=epsilon,
            knowledge_prior=prior,
            knowledge_weight=eta,
            penalty=penalty,
            penalty_mode="price",
            selected_action=1,
            logged_probability=float(policy[1]),
            reward=0.7,
            weight_floor=below_floor,
            timestep=1,
        ),
    ]
    result = run_go_no_go(
        selection,
        logs,
        baseline_value=0.6,
        ess_min_ratio=0.01,
        stability_delta=0.1,
        stability_window=5,
    )
    assert result.policy.ok
    assert not result.off_policy.ok
    assert not result.accepted
