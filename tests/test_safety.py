"""Tests for the primal–dual safety gate tuner."""

from __future__ import annotations

import math

import numpy as np

from semantic_lexicon.safety import (
    ConstraintSpec,
    GateBounds,
    ObjectiveSpec,
    SafetyGateTuner,
)


def test_primal_dual_tuner_solves_box_constrained_qp() -> None:
    parameter_names = ("x1", "x2")

    objective = ObjectiveSpec(
        function=lambda params: params["x1"] ** 2 + params["x2"] ** 2 - params["x1"] - params["x2"],
        gradient=lambda params: {
            "x1": 2.0 * params["x1"] - 1.0,
            "x2": 2.0 * params["x2"] - 1.0,
        },
    )

    constraints = [
        ConstraintSpec(
            name="linear",
            function=lambda params: params["x1"] + params["x2"] - 1.0,
            gradient=lambda params: {"x1": 1.0, "x2": 1.0},
        )
    ]

    bounds = {
        "x1": GateBounds(lower=0.0, upper=1.0),
        "x2": GateBounds(lower=0.0, upper=1.0),
    }

    tuner = SafetyGateTuner(
        objective,
        constraints,
        parameter_names,
        bounds=bounds,
        primal_step=0.2,
        dual_step=0.4,
        tolerance=1e-5,
        max_iterations=5000,
    )

    result = tuner.run({"x1": 0.2, "x2": 0.8})

    assert result.converged
    assert result.iterations <= tuner.max_iterations
    assert result.history
    assert len(result.history) == result.iterations

    x1 = result.parameters["x1"]
    x2 = result.parameters["x2"]
    assert math.isclose(x1, 0.5, abs_tol=3e-3)
    assert math.isclose(x2, 0.5, abs_tol=3e-3)

    residuals = np.asarray(list(result.residuals.values()), dtype=float)
    assert np.all(residuals <= 1e-4)

    # Dual feasibility and complementary slackness.
    dual = result.dual_variables["linear"]
    assert dual >= -1e-8
    assert math.isclose(dual * residuals[0], 0.0, abs_tol=1e-6)

    # Check that the projected box constraints are satisfied.
    assert 0.0 <= x1 <= 1.0
    assert 0.0 <= x2 <= 1.0
