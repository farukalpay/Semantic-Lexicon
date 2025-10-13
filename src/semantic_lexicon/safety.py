"""Projected primal–dual controller for safety gate parameters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np

ParameterVector = Mapping[str, float]


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification of the scalar objective used in gate tuning."""

    function: Callable[[ParameterVector], float]
    gradient: Callable[[ParameterVector], Mapping[str, float]] | None = None


@dataclass(frozen=True)
class ConstraintSpec:
    """A convex constraint of the form ``h(theta) <= 0``."""

    name: str
    function: Callable[[ParameterVector], float]
    gradient: Callable[[ParameterVector], Mapping[str, float]] | None = None


@dataclass(frozen=True)
class GateBounds:
    """Optional box constraints applied after each primal step."""

    lower: float | None = None
    upper: float | None = None


@dataclass
class GateTuningResult:
    """Result payload returned after running the primal–dual loop."""

    converged: bool
    iterations: int
    parameters: dict[str, float]
    dual_variables: dict[str, float]
    residuals: dict[str, float]
    history: list[dict[str, dict[str, float]]]


class SafetyGateTuner:
    """Projected primal–dual controller for auto-tuning safety gates.

    The controller minimises ``objective`` subject to the constraints encoded
    by ``constraints``. Parameters are stored in a deterministic order provided
    by ``parameter_names`` so the history can be interpreted in downstream
    diagnostics.
    """

    def __init__(
        self,
        objective: ObjectiveSpec,
        constraints: Sequence[ConstraintSpec],
        parameter_names: Sequence[str],
        *,
        bounds: Mapping[str, GateBounds] | None = None,
        primal_step: float = 1e-2,
        dual_step: float = 1e-2,
        tolerance: float = 1e-5,
        max_iterations: int = 1000,
    ) -> None:
        self.objective = objective
        self.constraints = list(constraints)
        self.parameter_names = list(parameter_names)
        self.bounds = dict(bounds or {})
        self.primal_step = float(primal_step)
        self.dual_step = float(dual_step)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

    # Public API -----------------------------------------------------------------
    def run(
        self,
        initial_parameters: Mapping[str, float],
        *,
        initial_dual: Mapping[str, float] | None = None,
    ) -> GateTuningResult:
        """Execute projected primal–dual updates until convergence.

        Parameters not specified in ``initial_parameters`` default to zero. Dual
        variables likewise default to zero when not provided.
        """

        theta = self._vectorise(initial_parameters)
        lambdas = self._dual_vector(initial_dual)
        history: list[dict[str, dict[str, float]]] = []

        for iteration in range(1, self.max_iterations + 1):
            mapping = self._to_mapping(theta)
            grad = self._lagrangian_gradient(mapping, lambdas)
            theta_next = theta - self.primal_step * grad
            theta_next = self._project(theta_next)

            mapping_next = self._to_mapping(theta_next)
            residuals = np.asarray(
                [spec.function(mapping_next) for spec in self.constraints],
                dtype=float,
            )
            lambdas_next = np.maximum(0.0, lambdas + self.dual_step * residuals)

            theta_delta = float(np.linalg.norm(theta_next - theta))
            positive_residual = np.maximum(residuals, 0.0)

            history.append(
                {
                    "parameters": mapping_next,
                    "residuals": {
                        spec.name: float(value) for spec, value in zip(self.constraints, residuals)
                    },
                    "dual": {
                        spec.name: float(value)
                        for spec, value in zip(self.constraints, lambdas_next)
                    },
                }
            )

            theta = theta_next
            lambdas = lambdas_next

            max_residual = float(np.max(positive_residual)) if positive_residual.size else 0.0
            if max_residual <= self.tolerance and theta_delta <= self.tolerance:
                return GateTuningResult(
                    converged=True,
                    iterations=iteration,
                    parameters=mapping_next,
                    dual_variables={
                        spec.name: float(value) for spec, value in zip(self.constraints, lambdas)
                    },
                    residuals={
                        spec.name: float(max(res, 0.0))
                        for spec, res in zip(self.constraints, residuals)
                    },
                    history=history,
                )

        mapping = self._to_mapping(theta)
        return GateTuningResult(
            converged=False,
            iterations=self.max_iterations,
            parameters=mapping,
            dual_variables={
                spec.name: float(value) for spec, value in zip(self.constraints, lambdas)
            },
            residuals={
                spec.name: float(max(spec.function(mapping), 0.0)) for spec in self.constraints
            },
            history=history,
        )

    # Internal helpers -----------------------------------------------------------
    def _vectorise(self, params: Mapping[str, float]) -> np.ndarray:
        vector = np.zeros(len(self.parameter_names), dtype=float)
        for idx, name in enumerate(self.parameter_names):
            vector[idx] = float(params.get(name, 0.0))
        return vector

    def _dual_vector(self, dual: Mapping[str, float] | None) -> np.ndarray:
        vector = np.zeros(len(self.constraints), dtype=float)
        if dual is None:
            return vector
        for idx, spec in enumerate(self.constraints):
            vector[idx] = max(0.0, float(dual.get(spec.name, 0.0)))
        return vector

    def _to_mapping(self, vector: np.ndarray) -> dict[str, float]:
        return {name: float(vector[idx]) for idx, name in enumerate(self.parameter_names)}

    def _lagrangian_gradient(self, params: Mapping[str, float], dual: np.ndarray) -> np.ndarray:
        grad_obj = self._gradient(self.objective, params)
        gradient = grad_obj.copy()
        for idx, spec in enumerate(self.constraints):
            grad_con = self._gradient(spec, params)
            gradient += float(dual[idx]) * grad_con
        return gradient

    def _gradient(
        self,
        spec: ObjectiveSpec | ConstraintSpec,
        params: Mapping[str, float],
        *,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        if spec.gradient is not None:
            grad_map = spec.gradient(params)
            return np.asarray(
                [float(grad_map.get(name, 0.0)) for name in self.parameter_names],
                dtype=float,
            )

        base = np.asarray(
            [float(params.get(name, 0.0)) for name in self.parameter_names],
            dtype=float,
        )
        grad = np.zeros_like(base)
        for idx in range(len(self.parameter_names)):
            delta = np.zeros_like(base)
            delta[idx] = epsilon
            plus_params = self._to_mapping(base + delta)
            minus_params = self._to_mapping(base - delta)
            forward = spec.function(plus_params)
            backward = spec.function(minus_params)
            grad[idx] = (forward - backward) / (2 * epsilon)
        return grad

    def _project(self, vector: np.ndarray) -> np.ndarray:
        clipped = np.asarray(vector, dtype=float)
        for idx, name in enumerate(self.parameter_names):
            bound = self.bounds.get(name)
            if bound is None:
                continue
            lower = bound.lower if bound.lower is not None else -np.inf
            upper = bound.upper if bound.upper is not None else np.inf
            clipped[idx] = float(np.clip(clipped[idx], lower, upper))
        return clipped


def run_primal_dual_autotune(
    objective: ObjectiveSpec,
    constraints: Iterable[ConstraintSpec],
    initial_parameters: Mapping[str, float],
    *,
    parameter_names: Sequence[str],
    bounds: Mapping[str, GateBounds] | None = None,
    primal_step: float = 1e-2,
    dual_step: float = 1e-2,
    tolerance: float = 1e-5,
    max_iterations: int = 1000,
) -> GateTuningResult:
    """Convenience wrapper for one-off primal–dual tuning runs."""

    tuner = SafetyGateTuner(
        objective,
        list(constraints),
        list(parameter_names),
        bounds=bounds,
        primal_step=primal_step,
        dual_step=dual_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return tuner.run(initial_parameters)
