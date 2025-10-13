# Intent-Bandit Analysis Toolkit

This appendix summarises the theoretical guarantees and the concrete helper
implementations that back the improved intent-selection loop. Each section maps
the specification work to the public helpers exposed from
`semantic_lexicon.analysis` and highlights the regression tests in
`tests/test_analysis.py` that keep the guarantees executable.

## 1. Composite Reward Engineering

Let \(R_c = \mathbf{1}\{a = a^*\}\),
\(R_{\text{conf}} = 1 - |p(a \mid s) - \mathbf{1}\{a = a^*\}|\),
\(R_{\text{sem}} = \operatorname{sim}(s, a)\), and
\(R_{\text{fb}} = \text{feedback}\), mirroring the
[`RewardComponents`](../src/semantic_lexicon/analysis/reward.py) dataclass. The
combined reward is
\[
R(s, a, a^*) = \sum_{i=1}^4 w_i R_i(s, a, a^*)
\]
with weights \(w \in \Delta^3\) constrained to the probability simplex. Because
each component is clipped into \([0, 1]\) by
[`RewardComponents.as_array`](../src/semantic_lexicon/analysis/reward.py) and the
weights are projected back onto the simplex by
[`project_to_simplex`](../src/semantic_lexicon/analysis/reward.py), the linear
combination remains bounded in \([0, 1]\). This satisfies the assumptions of
EXP3 (bounded rewards), so the standard regret guarantee applies without
modification. The helper
[`composite_reward_bound`](../src/semantic_lexicon/analysis/regret.py) provides an
explicit numerical bound and validates that the supplied weights satisfy the
simplex constraint.

### Optimal Weight Estimation

Given historical tuples \((R_t, y_t)\) with realised reward \(y_t\) and component
vector \(R_t\), the weight vector is obtained by solving
\[
\min_{w \in \Delta^3} \sum_t (w^\top R_t - y_t)^2.
\]
The programme is strictly convex, hence the optimum is the projection of the
unconstrained least-squares solution onto \(\Delta^3\). The implementation in
[`estimate_optimal_weights`](../src/semantic_lexicon/analysis/reward.py) solves the
regularised normal equations before calling `project_to_simplex`, which uses the
efficient algorithm of Wang & Carreira-Perpiñán (2013). The regression fixture
`tests/test_analysis.py::test_estimate_optimal_weights_simple_case` reconstructs
historical rewards with the returned weights to guard this behaviour.

## 2. Bayesian Confidence Calibration

Intent probabilities are modelled with a Dirichlet prior \(\operatorname{Dir}(\alpha)\).
After observing counts \(n_i\) for each intent, the posterior parameters are
\(\alpha_i + n_i\), yielding the posterior predictive distribution
\(p(i \mid D) = (\alpha_i + n_i)/(\sum_j \alpha_j + \sum_j n_j)\). Under the
absolute-error loss used by the expected calibration error (ECE), this posterior
mean minimises the Bayes risk. The shrinkage-based calibrator implemented in
[`DirichletCalibrator`](../src/semantic_lexicon/analysis/calibration.py) mixes the
model probabilities with the posterior predictive mean via
\(\lambda = \frac{\sum_j n_j}{\sum_j n_j + \sum_j \alpha_j}\), ensuring that
confidence scores converge towards calibrated frequencies as data accumulates.

Because the posterior predictive is the Bayes estimator for the absolute-error
loss, substituting it into
\(\operatorname{ECE} = \mathbb{E}[|\mathbb{P}(\hat{Y} = Y \mid \hat{p} = p) - p|]\)
minimises the expectation. Consequently, the calibrated scores reduce ECE and
stabilise the reward term \(R_{\text{conf}}\). The unit test
`tests/test_analysis.py::test_dirichlet_calibration_reduces_ece` exercises this
loop by confirming that the calibrated probability is closer to the empirical
accuracy.

## 3. EXP3 Regret Bound with Composite Rewards

The EXP3 update remains unchanged because the composite reward stays in
\([0, 1]\). Standard analysis (Auer et al., 2002) therefore yields
\[
\mathbb{E}[R_T] \le 2.63 \sqrt{K T \log K}
\]
for \(K = 4\) intents. The simulation helper
[`simulate_intent_bandit`](../src/semantic_lexicon/analysis/regret.py) evaluates this
bound numerically. It converts component rewards into scalars with
[`composite_reward`](../src/semantic_lexicon/analysis/reward.py) and records the
regret trajectory. `tests/test_analysis.py::test_exp3_regret_simulation_bounded`
ensures that the simulated regret stays below
[`exp3_expected_regret`](../src/semantic_lexicon/analysis/regret.py) when run on the
bundled sample data.

## 4. Systematic Error Correction via SVD

Let \(C\) denote the intent confusion matrix. The correction matrix that
minimises \(\lVert C T - I \rVert_F\) is the Moore–Penrose pseudoinverse
\(C^+\), obtained through the singular value decomposition
\(C = U \Sigma V^\top\) as \(T = V \Sigma^+ U^\top\) with damped reciprocals of
the singular values. Applying the correction right-multiplies the confusion
matrix by \(C^+\), contracting systematic biases. The helper
[`compute_confusion_correction`](../src/semantic_lexicon/analysis/error.py) performs
this computation and [`confusion_correction_residual`](../src/semantic_lexicon/analysis/error.py)
reports the Frobenius residual. The regression guard
`tests/test_analysis.py::test_confusion_correction_reduces_residual` checks that the
residual decreases after applying the learned transform.

## 5. Convergence of the Coupled System

The joint classifier-bandit dynamics can be cast as the stochastic approximation
\(\theta_{t+1} = \theta_t + \gamma_t H(\theta_t, X_t)\) where the Robbins–Monro
step sizes \(\gamma_t\) satisfy \(\sum_t \gamma_t = \infty\) and
\(\sum_t \gamma_t^2 < \infty\). Under Lipschitz continuity of the operator and
bounded noise variance, classical results guarantee almost-sure convergence to a
stable fixed point. The scaffold provided by
[`RobbinsMonroProcess`](../src/semantic_lexicon/analysis/convergence.py) mirrors this
setup for experimentation, while
[`convergence_rate_bound`](../src/semantic_lexicon/analysis/convergence.py) encodes the
resulting \(O(1/\sqrt{n})\) sample-complexity bound on the estimation error. The
unit test `tests/test_analysis.py::test_robbins_monro_convergence_rate` verifies that
the deterministic trajectory shrinks and that the bound stays positive for
representative parameters.

These utilities integrate with the training scripts and experiments, providing a
transparent and verifiable foundation for ethical intent selection.

## 6. Primal–Dual Safety Gates

Operational safeguards such as rule-gap limits, minimum selection probability
floors, effective sample size (ESS) targets, fairness thresholds, and stability
conditions can be encoded as convex constraints \(h_m(\theta) \le 0\). The
augmented objective for the constrained programme is the dual Lagrangian
\[
\mathcal{L}(\theta, \lambda) = f(\theta) + \sum_m \lambda_m \, h_m(\theta),
\quad \lambda_m \ge 0,
\]
where \(f\) denotes the scalar gate-loss surrogate we wish to minimise. The
projected primal–dual method used in `semantic_lexicon.safety` mirrors the
textbook iteration:
\[
\theta_{t+1} = \Pi_{\Theta}\bigl(\theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t, \lambda_t)\bigr), \qquad
\lambda_{m, t+1} = \big[\lambda_{m, t} + \rho\, h_m(\theta_{t+1})\big]_+.
\]
The primal update takes a gradient descent step on the Lagrangian before
projecting onto the feasible box \(\Theta\) (if present), while the dual update
ascends on the constraint violation produced by the fresh primal iterate. A
Lyapunov function composed of the squared positive residuals and the dual error,
\[
V_t = \tfrac{1}{2} \sum_m h_m^+(\theta_t)^2 + \tfrac{1}{2}
\lVert \lambda_t - \lambda^* \rVert^2,
\]
monotonically decreases under these dynamics, i.e. \(V_{t+1} \le V_t\), which
guarantees that the constraint residuals converge to zero. In practice this loop
automatically tunes the step sizes and gate parameters \(\{\eta, \tau_g, \alpha,
\mu, \beta, \rho\}\) until each safety gate reaches zero residual, providing a
robust template for satisfying multiple operational requirements simultaneously.

## 7. Consistency Check for a Three-Proposition Logic

To illustrate how the specification reacts to contradictory rule sets, consider
the propositional system with statements \(X, Y, Z\) and axioms
\(X \Rightarrow Y\), \(Y \Rightarrow Z\), and \(\neg Z\). A formal inference
shows that both \(X\) and \(Y\) must be false.

1. The implication \(Y \Rightarrow Z\) together with \(\neg Z\) yields
   \(\neg Y\) by **modus tollens**: if \(Y\) were true, then so would \(Z\),
   contradicting the axiom \(\neg Z\).
2. The implication \(X \Rightarrow Y\) and the newly derived \(\neg Y\) again
   give \(\neg X\) via modus tollens: a true \(X\) would force \(Y\) to be
   true, contradicting \(\neg Y\).

Consequently the only models of the axiom set assign the value “false” to both
\(X\) and \(Y\), while \(Z\) is explicitly false. This minimal derivation uses
standard rules of inference and doubles as a template for verifying logical
consistency in the tooling that monitors specification documents.
