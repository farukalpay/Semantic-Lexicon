"""Go/No-Go validation for AGENTS-aligned knowledge and policies."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SelectionSpec:
    """Specification of a knowledge selection and its group constraints."""

    selected: Sequence[str]
    group_membership: Mapping[str, Sequence[str]]
    group_bounds: Mapping[str, tuple[float | None, float | None]]
    knowledge: KnowledgeSignals | None = None


@dataclass(frozen=True)
class KnowledgeSignals:
    """Knowledge metrics used by the Go/No-Go knowledge gate."""

    coverage: float
    cohesion: float
    knowledge_raw: float
    knowledge_calibrated: float
    baseline_coverage: float
    baseline_cohesion: float
    baseline_knowledge_calibrated_median: float


@dataclass(frozen=True)
class GroupCheckResult:
    """Outcome of the hard constraint feasibility test."""

    feasible: bool
    violations: Mapping[str, tuple[int, int]]


@dataclass(frozen=True)
class PolicyLogEntry:
    """Logged policy step used for the policy and OPE checks."""

    agent: str
    scores: Sequence[float]
    temperature: float
    epsilon: float
    knowledge_prior: Sequence[float]
    knowledge_weight: float
    penalty: Sequence[float] | None
    penalty_mode: Literal["price", "congestion"]
    selected_action: int
    logged_probability: float
    reward: float
    weight_floor: float | None = None
    metrics: Sequence[float] | None = None
    timestep: int | None = None


@dataclass(frozen=True)
class FairnessConfig:
    """Configuration for fairness checks in the Go/No-Go test."""

    action_target: Sequence[float] | None = None
    action_tolerance: float | Sequence[float] = 0.0
    metric_target: Sequence[float] | None = None
    metric_tolerance: float = 0.0


@dataclass
class PolicyCheckResult:
    """Summary of the policy consistency test."""

    ok: bool
    mode: Literal["price", "congestion"] | None
    min_probability: float
    min_floor: float
    policies: tuple[FloatArray, ...] = field(default_factory=tuple)
    floors: tuple[float, ...] = field(default_factory=tuple)


@dataclass
class OffPolicyResult:
    """Summary of the off-policy value and fairness checks."""

    ok: bool
    snips: float
    ess: float
    lower_confidence_bound: float
    fairness_ok: bool
    fairness_gaps: Mapping[str, float] = field(default_factory=dict)
    ess_threshold: float = 0.0


@dataclass
class StabilityCheckResult:
    """Assessment of price or congestion stability."""

    ok: bool
    threshold: float
    window: int
    max_delta: float


@dataclass
class KnowledgeCheckResult:
    """Assessment of the knowledge lift conditions."""

    ok: bool
    coverage_delta: float
    cohesion_delta: float
    knowledge_calibrated: float
    knowledge_median: float


@dataclass
class GoNoGoResult:
    """Composite Go/No-Go decision with detailed diagnostics."""

    selection: GroupCheckResult
    policy: PolicyCheckResult
    off_policy: OffPolicyResult
    stability: StabilityCheckResult
    knowledge: KnowledgeCheckResult

    @property
    def accepted(self) -> bool:
        return (
            self.selection.feasible
            and self.policy.ok
            and self.off_policy.ok
            and self.stability.ok
            and self.knowledge.ok
        )


def _coerce_count(value: float | None, default: int) -> int:
    if value is None:
        return default
    value_float = float(value)
    if math.isnan(value_float):  # pragma: no cover - defensive
        return default
    return int(math.floor(value_float))


def _check_group_bounds(spec: SelectionSpec) -> GroupCheckResult:
    selection = [str(item) for item in spec.selected]
    membership: dict[str, tuple[str, ...]] = {}
    for concept, groups in spec.group_membership.items():
        cleaned = tuple(str(group) for group in groups if str(group))
        membership[str(concept)] = cleaned
    counts: dict[str, int] = {group: 0 for group in spec.group_bounds}
    for concept in selection:
        groups = membership.get(concept, ())
        for group in groups:
            if group in counts:
                counts[group] = counts.get(group, 0) + 1
    total = len(selection)
    violations: dict[str, tuple[int, int]] = {}
    feasible = True
    for group, (lower, upper) in spec.group_bounds.items():
        actual = counts.get(group, 0)
        lower_bound = _coerce_count(lower, 0)
        if lower is not None and 0 < lower < 1:
            lower_bound = int(math.ceil(lower * total))
        upper_bound = _coerce_count(upper, total)
        if upper is not None and 0 < upper < 1:
            upper_bound = int(math.floor(upper * total))
        lower_violation = max(0, lower_bound - actual)
        upper_violation = max(0, actual - upper_bound) if upper is not None else 0
        violations[group] = (lower_violation, upper_violation)
        if lower_violation > 0 or upper_violation > 0:
            feasible = False
    return GroupCheckResult(feasible=feasible, violations=violations)


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = np.asarray(logits - np.max(logits), dtype=float)
    exps = np.asarray(np.exp(shifted), dtype=float)
    denom = float(np.sum(exps)) or 1.0
    return cast(FloatArray, np.asarray(exps / denom, dtype=float))


def _policy_distribution(entry: PolicyLogEntry) -> tuple[FloatArray, float]:
    scores = np.asarray(entry.scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        msg = "Scores must be a one-dimensional non-empty sequence"
        raise ValueError(msg)
    penalty = np.zeros_like(scores)
    if entry.penalty is not None:
        penalty = np.asarray(entry.penalty, dtype=float)
        if penalty.shape != scores.shape:
            msg = "Penalty vector must match score dimensionality"
            raise ValueError(msg)
    prior = np.asarray(entry.knowledge_prior, dtype=float)
    if prior.shape != scores.shape:
        msg = "Knowledge prior must align with scores"
        raise ValueError(msg)
    if not (0.0 <= entry.epsilon <= 1.0):
        msg = "Epsilon must lie in [0, 1]"
        raise ValueError(msg)
    if not (0.0 <= float(entry.knowledge_weight) <= 1.0):
        msg = "Knowledge weight must lie in [0, 1]"
        raise ValueError(msg)
    temperature = max(float(entry.temperature), 1e-6)
    logits = np.asarray(
        (scores - penalty + entry.knowledge_weight * prior) / temperature,
        dtype=float,
    )
    base = _softmax(cast(FloatArray, logits))
    K = float(scores.size)
    policy = (1.0 - entry.epsilon) * base + entry.epsilon / K
    floor = entry.epsilon / K
    return cast(FloatArray, np.asarray(policy, dtype=float)), float(floor)


def _aggregate_metrics(
    logs: Sequence[PolicyLogEntry],
    policies: Sequence[FloatArray],
    floors: Sequence[float],
    fairness: FairnessConfig | None,
    baseline_value: float,
    ess_min_ratio: float,
) -> OffPolicyResult:
    ess_min = ess_min_ratio * len(logs)
    if not logs:
        return OffPolicyResult(
            ok=False,
            snips=0.0,
            ess=0.0,
            lower_confidence_bound=float("-inf"),
            fairness_ok=False,
            fairness_gaps={},
            ess_threshold=ess_min,
        )
    if len(policies) != len(logs) or len(floors) != len(logs):
        return OffPolicyResult(
            ok=False,
            snips=0.0,
            ess=0.0,
            lower_confidence_bound=float("-inf"),
            fairness_ok=False,
            fairness_gaps={},
            ess_threshold=ess_min,
        )
    rewards = np.asarray([float(entry.reward) for entry in logs], dtype=float)
    selected_indices = np.asarray([int(entry.selected_action) for entry in logs], dtype=int)
    logged_probs = np.asarray([float(entry.logged_probability) for entry in logs], dtype=float)
    weights: list[float] = []
    for idx, entry in enumerate(logs):
        policy = policies[idx]
        action = selected_indices[idx]
        denom_floor = entry.weight_floor if entry.weight_floor is not None else floors[idx]
        exploration_floor = floors[idx]
        if denom_floor + 1e-9 < exploration_floor:
            return OffPolicyResult(
                ok=False,
                snips=0.0,
                ess=0.0,
                lower_confidence_bound=float("-inf"),
                fairness_ok=False,
                fairness_gaps={},
                ess_threshold=ess_min,
            )
        denom = max(logged_probs[idx], denom_floor, 1e-12)
        weights.append(float(policy[action] / denom))
    weight_array = np.asarray(weights, dtype=float)
    numerator = float(np.sum(weight_array * rewards))
    denom = float(np.sum(weight_array))
    snips = numerator / denom if denom > 0 else 0.0
    ess_denom = float(np.sum(weight_array**2)) or 1.0
    ess = (denom**2) / ess_denom
    centered = rewards - snips
    variance = float(np.sum((weight_array * centered) ** 2))
    variance /= (denom**2) if denom > 0 else 1.0
    standard_error = math.sqrt(max(variance, 0.0) / max(len(logs), 1))
    lcb = snips - baseline_value - 1.96 * standard_error
    agent_counts: dict[str, Counter[int]] = {}
    agent_metrics: dict[str, list[np.ndarray]] = {}
    for entry in logs:
        agent_counts.setdefault(entry.agent, Counter()).update({entry.selected_action: 1})
        if entry.metrics is not None:
            agent_metrics.setdefault(entry.agent, []).append(np.asarray(entry.metrics, dtype=float))
    fairness_gaps: dict[str, float] = {}
    fairness_ok = True
    if fairness is not None:
        if fairness.action_target is not None:
            target = np.asarray(fairness.action_target, dtype=float)
            tolerance = fairness.action_tolerance
            if isinstance(tolerance, Sequence) and not isinstance(tolerance, (str, bytes)):
                tol_vec = np.asarray(tolerance, dtype=float)
                if tol_vec.shape != target.shape:
                    msg = "Action tolerance must match action target dimensionality"
                    raise ValueError(msg)
            else:
                tol_vec = np.full(target.shape, float(tolerance), dtype=float)
            for agent, counts in agent_counts.items():
                total = sum(counts.values())
                if total == 0:
                    continue
                distribution = np.zeros_like(target)
                for action, count in counts.items():
                    if 0 <= action < target.size:
                        distribution[action] = count / total
                diff = np.abs(distribution - target) - tol_vec
                deviation = float(np.max(np.maximum(diff, 0.0)))
                fairness_gaps[agent] = deviation
                if deviation > 0:
                    fairness_ok = False
        if fairness.metric_target is not None:
            target_vec = np.asarray(fairness.metric_target, dtype=float)
            for agent, metrics in agent_metrics.items():
                if not metrics:
                    continue
                averaged = np.mean(np.stack(metrics, axis=0), axis=0)
                if averaged.shape != target_vec.shape:
                    msg = "Metric dimension mismatch in fairness check"
                    raise ValueError(msg)
                gap = float(np.max(np.abs(averaged - target_vec)))
                deviation = max(0.0, gap - fairness.metric_tolerance)
                if agent in fairness_gaps:
                    fairness_gaps[agent] = max(fairness_gaps[agent], deviation)
                else:
                    fairness_gaps[agent] = deviation
                if deviation > 0:
                    fairness_ok = False
    ok = ess >= ess_min and lcb >= 0.0 and fairness_ok
    return OffPolicyResult(
        ok=ok,
        snips=snips,
        ess=ess,
        lower_confidence_bound=lcb,
        fairness_ok=fairness_ok,
        fairness_gaps=fairness_gaps,
        ess_threshold=ess_min,
    )


def _check_policy(logs: Sequence[PolicyLogEntry]) -> PolicyCheckResult:
    if not logs:
        return PolicyCheckResult(
            ok=False,
            mode=None,
            min_probability=0.0,
            min_floor=0.0,
        )
    modes = {entry.penalty_mode for entry in logs}
    if len(modes) != 1:
        return PolicyCheckResult(
            ok=False,
            mode=None,
            min_probability=0.0,
            min_floor=0.0,
        )
    mode = modes.pop()
    min_probability = float("inf")
    min_floor = float("inf")
    policies: list[FloatArray] = []
    floors: list[float] = []
    for entry in logs:
        try:
            policy, floor = _policy_distribution(entry)
        except ValueError:
            return PolicyCheckResult(
                ok=False,
                mode=mode,
                min_probability=0.0,
                min_floor=0.0,
            )
        policies.append(policy)
        floors.append(floor)
        min_probability = min(min_probability, float(np.min(policy)))
        min_floor = min(min_floor, floor)
        if min_probability + 1e-9 < floor:
            return PolicyCheckResult(
                ok=False,
                mode=mode,
                min_probability=min_probability,
                min_floor=floor,
            )
    return PolicyCheckResult(
        ok=True,
        mode=mode,
        min_probability=min_probability,
        min_floor=min_floor,
        policies=tuple(policies),
        floors=tuple(floors),
    )


def _check_stability(
    logs: Sequence[PolicyLogEntry],
    policy_result: PolicyCheckResult,
    *,
    stability_delta: float,
    stability_window: int,
) -> StabilityCheckResult:
    if not logs or not policy_result.ok:
        return StabilityCheckResult(
            ok=False,
            threshold=stability_delta,
            window=0,
            max_delta=float("inf"),
        )
    penalty_groups: dict[int, list[np.ndarray]] = {}
    for idx, entry in enumerate(logs):
        if entry.penalty is None:
            return StabilityCheckResult(
                ok=False,
                threshold=stability_delta,
                window=0,
                max_delta=float("inf"),
            )
        penalty = np.asarray(entry.penalty, dtype=float)
        if penalty.ndim != 1:
            msg = "Penalty vector must be one-dimensional for stability checks"
            raise ValueError(msg)
        step_key = entry.timestep if entry.timestep is not None else idx
        bucket = penalty_groups.setdefault(int(step_key), [])
        bucket.append(penalty)
    if not penalty_groups:
        return StabilityCheckResult(
            ok=True,
            threshold=stability_delta,
            window=0,
            max_delta=0.0,
        )
    ordered_steps = sorted(penalty_groups)
    aggregated: list[np.ndarray] = []
    for step in ordered_steps:
        vectors = penalty_groups[step]
        shapes = {vec.shape for vec in vectors}
        if len(shapes) != 1:
            msg = "Penalty vectors must share a common dimensionality"
            raise ValueError(msg)
        aggregated.append(np.mean(np.stack(vectors, axis=0), axis=0))
    if len(aggregated) < 2:
        return StabilityCheckResult(
            ok=True,
            threshold=stability_delta,
            window=0,
            max_delta=0.0,
        )
    deltas: list[float] = []
    for prev, nxt in zip(aggregated, aggregated[1:]):
        delta = float(np.sum(np.abs(nxt - prev)))
        deltas.append(delta)
    window = min(len(deltas), max(stability_window, 1))
    recent = deltas[-window:] if window > 0 else deltas
    max_delta = max(recent) if recent else 0.0
    ok = max_delta <= stability_delta
    return StabilityCheckResult(
        ok=ok,
        threshold=stability_delta,
        window=window,
        max_delta=max_delta,
    )


def _check_knowledge(selection: SelectionSpec) -> KnowledgeCheckResult:
    signals = selection.knowledge
    if signals is None:
        return KnowledgeCheckResult(
            ok=False,
            coverage_delta=float("nan"),
            cohesion_delta=float("nan"),
            knowledge_calibrated=float("nan"),
            knowledge_median=float("nan"),
        )
    coverage_delta = float(signals.coverage - signals.baseline_coverage)
    cohesion_delta = float(signals.cohesion - signals.baseline_cohesion)
    knowledge_cal = float(signals.knowledge_calibrated)
    knowledge_median = float(signals.baseline_knowledge_calibrated_median)
    tolerance = 1e-9
    ok = (
        coverage_delta >= -tolerance
        and cohesion_delta >= -tolerance
        and knowledge_cal + tolerance >= knowledge_median
    )
    return KnowledgeCheckResult(
        ok=ok,
        coverage_delta=coverage_delta,
        cohesion_delta=cohesion_delta,
        knowledge_calibrated=knowledge_cal,
        knowledge_median=knowledge_median,
    )


def run_go_no_go(
    selection: SelectionSpec,
    logs: Sequence[PolicyLogEntry],
    *,
    fairness: FairnessConfig | None = None,
    baseline_value: float = 0.0,
    ess_min_ratio: float = 0.01,
    stability_delta: float = 0.1,
    stability_window: int = 10,
) -> GoNoGoResult:
    """Execute the composite Go/No-Go evaluation."""

    selection_result = _check_group_bounds(selection)
    policy_result = _check_policy(logs)
    stability_result = _check_stability(
        logs,
        policy_result,
        stability_delta=stability_delta,
        stability_window=stability_window,
    )
    off_policy_result = _aggregate_metrics(
        logs,
        list(policy_result.policies),
        list(policy_result.floors),
        fairness,
        baseline_value,
        ess_min_ratio,
    )
    knowledge_result = _check_knowledge(selection)
    return GoNoGoResult(
        selection=selection_result,
        policy=policy_result,
        off_policy=off_policy_result,
        stability=stability_result,
        knowledge=knowledge_result,
    )
