# Intent-Bandit Analysis Toolkit

This appendix summarises the theoretical guarantees and practical tooling that
back the improved intent-selection loop. Each section corresponds to one of the
requested analysis tasks and references the concrete helpers shipped in
`semantic_lexicon.analysis`.

## 1. Composite Reward Engineering

Let \(R_c = \mathbf{1}\{a = a^*\}\), \(R_{\text{conf}} = 1 - |p(a \mid s) - \mathbf{1}\{a = a^*\}|\),
\(R_{\text{sem}} = \operatorname{sim}(s, a)\), and \(R_{\text{fb}} = \text{feedback}\).
The combined reward is
\[
R(s, a, a^*) = \sum_{i=1}^4 w_i R_i(s, a, a^*)
\]
with weights \(w \in \Delta^3\) constrained to the probability simplex. Because
each component is clipped into \([0, 1]\), the linear combination is bounded in
\([0, 1]\). This satisfies the assumptions of EXP3 (bounded rewards) and the
existing regret guarantee applies without modification. The helper
[`composite_reward_bound`](../src/semantic_lexicon/analysis/regret.py) enforces the
simplex constraint programmatically.

### Optimal Weight Estimation

Given historical tuples \((R_t, y_t)\) with realised reward \(y_t\) and component
vector \(R_t\), the weight vector is obtained by solving
\[
\min_{w \in \Delta^3} \sum_t (w^\top R_t - y_t)^2.
\]
The programme is strictly convex, hence the optimum is the projection of the
unconstrained least-squares solution onto \(\Delta^3\). The implementation is
provided by [`estimate_optimal_weights`](../src/semantic_lexicon/analysis/reward.py)
which uses the efficient projection algorithm of Wang & Carreira-Perpiñán
(2013).

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
stabilise the reward term \(R_{\text{conf}}\).

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
regret trajectory. The tests in `tests/test_analysis.py` verify that the bound
holds on the bundled sample data.

## 4. Systematic Error Correction via SVD

Let \(C\) denote the intent confusion matrix. The correction matrix that
minimises \(\lVert C T - I \rVert_F\) is the Moore–Penrose pseudoinverse
\(C^+\), obtained through the singular value decomposition
\(C = U \Sigma V^\top\) as \(T = V \Sigma^+ U^\top\) with damped reciprocals of
the singular values. Applying the correction right-multiplies the confusion
matrix by \(C^+\), contracting systematic biases. The helper
[`compute_confusion_correction`](../src/semantic_lexicon/analysis/error.py) performs
this computation and [`confusion_correction_residual`](../src/semantic_lexicon/analysis/error.py)
reports the Frobenius residual, enabling regression tests that ensure the
correction reduces structured errors.

## 5. Convergence of the Coupled System

The joint classifier-bandit dynamics can be cast as the stochastic approximation
\(\theta_{t+1} = \theta_t + \gamma_t H(\theta_t, X_t)\) where the Robbins–Monro
step sizes \(\gamma_t\) satisfy \(\sum_t \gamma_t = \infty\) and
\(\sum_t \gamma_t^2 < \infty\). Under Lipschitz continuity of the operator and
bounded noise variance, classical results guarantee almost-sure convergence to a
stable fixed point. The scaffold provided by
[`RobbinsMonroProcess`](../src/semantic_lexicon/analysis/convergence.py) mirrors this
setup for experimentation, while [`convergence_rate_bound`](../src/semantic_lexicon/analysis/convergence.py)
encodes the resulting \(O(1/\sqrt{n})\) sample-complexity bound on the estimation
error.

These utilities integrate with the training scripts and experiments, providing a
transparent and verifiable foundation for ethical intent selection.
