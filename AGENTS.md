# Multi-Agent Online Intent Prediction — Spec v3 (Math-First, No Code)
_Clarifies \(s_\theta\), adds \(\delta^{(i)}\) regularization, Boltzmann exploration, precise reward modeling, stable price dynamics, explicit fairness metrics, consistent aggregation, and stronger off-policy evaluation (OPE)._

## 0) Notation and Roles of \(s_\theta\)
- Agents \(\mathcal{A}=\{1,\dots,M\}\), intents \(\{1,\dots,K\}\), vocab size \(V\).
- Context \(x\), BoW vector \(X\in\mathbb{R}^V\).
- Parameters \(\theta=(W,b)\), with \(W\in\mathbb{R}^{V\times K},\ b\in\mathbb{R}^K\).
- Per-agent class-wise bias \(\delta^{(i)}\in\mathbb{R}^K\). **Scores**
  \[
  s_\theta^{(i)}(X) = X^\top W + b + \delta^{(i)}\ \in \mathbb{R}^K.
  \]
**Two uses of \(s_\theta\):**
1. **Reward regression:** \(s_\theta(X)_a \approx \mathbb{E}[\,r\mid X,a\,]\) (bandit value). Train via IS/DR regression (Sec. 3).
2. **Policy logits:** \(\pi_\theta(a\mid x)\propto \exp(s_\theta(X)_a/\tau)\) (Boltzmann). Use \(\epsilon\)-mixture or pure Softmax (Sec. 1).
> You may **split heads**: value head \(s_\theta\) for reward regression; policy head \(g_\theta\) for logits. For the **tied-head** variant set \(g_\theta\equiv s_\theta\).

**Regularization/constraints on \(\delta^{(i)}\):**
\(
\sum_{i=1}^M \delta^{(i)}=0\) (centered) and penalty \( \mu_\delta \sum_i \|\delta^{(i)}\|_2^2\) (or group-lasso \( \mu_{\text{grp}}\sum_i\|\delta^{(i)}\|_2\)). Projection step each round: \( \delta^{(i)}\leftarrow \delta^{(i)}-\frac{1}{M}\sum_j \delta^{(j)}\).

---

## 1) Policies: \(\epsilon\)-Mixture and Boltzmann (Softmax) Exploration
**Price mode (no congestion term):**
\[
\pi_\theta^{(i)}(a\mid x)=(1-\epsilon_t^{(i)})\,\mathrm{Softmax}\!\left(\frac{s_\theta^{(i)}(X)-\lambda_t}{\tau_t}\right)_a + \epsilon_t^{(i)}\frac{1}{K}.
\]

**Congestion-penalty mode (no prices):**
\[
\pi_\theta^{(i)}(a\mid x)=(1-\epsilon_t^{(i)})\,\mathrm{Softmax}\!\left(\frac{s_\theta^{(i)}(X)-c\,\hat n_t}{\tau_t}\right)_a + \epsilon_t^{(i)}\frac{1}{K}.
\]

- **Do not combine** penalties: choose one.
- **Schedules:** \( \epsilon_t^{(i)}=\epsilon_0+\min(1,t/D)(\epsilon_1-\epsilon_0)\), \( \tau_t=\tau_0+\min(1,t/D_\tau)(\tau_1-\tau_0)\).
- **Logging:** record \(p=\pi_\theta^{(i)}(a\mid x)\) at selection time for replay/OPE.

---

## 2) Rewards: Model and Transformations
Potential outcomes \(Y(a)\in[0,1]\) with noise \(\xi\) (sub-Gaussian). Observed \(r=Y(a_t)+\xi\).  
Reward shaping \(\tilde r = \phi(r)\) with monotone \(\phi\) (e.g., identity, \(\mathrm{logit}^{-1}\), clipping) is allowed; be consistent across training and OPE.

- **Binary** rewards: GLM/logistic bandit likelihood is natural.
- **Real-valued** rewards: squared loss on \(\tilde r\).  
- **Baselines:** variance reduction via \(r\leftarrow r-b\), where \(b\) is EMA or critic value \(V\).

Team reward: \(R_t=\sum_i r_t^{(i)}\). Difference rewards: \(D_t^{(i)}=G(z_t)-G(z_t^{(-i)}\cup\{c_0\})\).

---

## 3) Contextual Bandit Training (IS/DR)
Each logged tuple stores \((X,a,r,p)\). With \(\epsilon\)-mixture policy, \(p\ge \epsilon/K\).

- **IS regression (clipped):**
\[
\tilde{L}_{\mathrm{IS}}=\frac{1}{|B|}\sum_{(X,a,r,p)\in B}\frac{\big( s_\theta(X)_a - r \big)^2}{\max(p,\epsilon)}+\frac{\lambda}{2}\|W\|_F^2+\frac{\lambda}{2}\|b\|_2^2+\mu_\delta\sum_i\|\delta^{(i)}\|_2^2.
\]

- **DR regression (requires reward model \(\hat r\)):**
\[
\tilde{L}_{\mathrm{DR}}=\frac{1}{|B|}\sum_{(X,a,r,p)\in B}\Big[\big(s_\theta(X)_a-\hat r(X,a)\big)^2+\frac{(r-\hat r(X,a))^2}{\max(p,\epsilon)}\Big]+\text{reg}.
\]

For **binary** \(r\): replace squared loss with negative log-likelihood, keeping the IS/DR weighting.

---

## 4) Critics, Values, Advantages
Centralized/shared critic \(Q_\psi(x,a)\approx \mathbb{E}[r\mid x,a]\). Then
\[
V_\psi(x)=\sum_{a'} \pi_\theta(a'\mid x)\,Q_\psi(x,a'),\qquad 
A_\psi(x,a)=Q_\psi(x,a)-V_\psi(x).
\]
Policy-gradient (per agent \(i\)):
\[
\nabla_\theta J \approx \mathbb{E}\!\left[(r_t^{(i)}-V_\psi(x_t^{(i)}))\,\nabla_\theta\log\pi_\theta^{(i)}(a_t^{(i)}\mid x_t^{(i)})\right].
\]
If tying \(Q_\psi\equiv s_\theta\), keep a target network or slow updates to avoid bias.

---

## 5) Coordination via Prices or Load Forecasts
**Capacities** \(\mathrm{cap}\in\mathbb{R}_+^K\), realized loads \(n_{t,a}=\sum_i \mathbf{1}\{a_t^{(i)}=a\}\).

- **Prices (dual ascent with damping):**
\[
\lambda_{t+1,a}=(1-\beta)\lambda_{t,a}+\beta\,\Big[\lambda_{t,a}+\rho\,\big(n_{t,a}-\mathrm{cap}_a\big)\Big]_+,
\]
with \(0<\beta\le 1\) and step \(0<\rho<\rho_{\max}\) for stability (choose \(\rho\) s.t. oscillations damp with observed delays).

- **Load forecast (EMA) for congestion penalties:**
\[
\hat n_{t+1,a}=\alpha \hat n_{t,a}+(1-\alpha) n_{t,a},\quad \alpha\in[0,1).
\]

> Choose **either** price mode (\(\lambda\)) **or** congestion-penalty mode (\(c\,\hat n\)).

---

## 6) Aggregation (Consistent Formulas)
- **Centralized (gradient deltas):**
\[
\theta_{t+1}=\theta_t+\sum_{i=1}^M w_i\,\Delta\theta_t^{(i)},\quad \sum_i w_i=1,\quad w_i\propto |B^{(i)}|\,e^{-\gamma\,\text{staleness}^{(i)}}.
\]
- **FedAvg (model averaging):**
\[
\theta_{t+1}=\sum_{i=1}^M p_i\,\theta_t^{(i,S)},\quad \sum_i p_i=1,\quad p_i=\frac{n_i}{\sum_j n_j}.
\]
- **FedProx:** local objective \(\tilde{L}^{(i)}+\frac{\mu}{2}\|\theta-\theta_t\|^2\), then FedAvg.
- **Decentralized consensus:** \( \theta^{(i)}\leftarrow \sum_j P_{ij}\,\theta^{(j)} \) with doubly-stochastic \(P\) (spectral gap \(\gamma_P\)).

---

## 7) Fairness Metrics and Constraints
Choose KPI \(\phi^{(i)}\) and estimator \(\widehat\phi^{(i)}_t\) (EMA or sliding window). Common choices:
- **Hit-rate (accuracy):** \(\phi^{(i)}=\mathbb{E}[\mathbf{1}\{\hat y=y\}]\).
- **Mean reward:** \(\phi^{(i)}=\mathbb{E}[r^{(i)}]\).
- **Coverage per intent:** \(q_{i,a}=\Pr\{a_t^{(i)}=a\}\); enforce \( |q_{i,a}-\alpha_a|\le \epsilon_a\).
- **Calibration drift:** \(\mathrm{Cal}^{(i)}=\mathbb{E}\big[|\,\sigma(s_\theta^{(i)}(X)_{\hat a})-r\,|\big]\) for \(\sigma=\mathrm{sigmoid}\) or Softmax prob.

Constraints via dual ascent:
\[
\nu_i \leftarrow \Big[\nu_i + \eta_\nu\,(u_i - \widehat{\phi}^{(i)}_t)\Big]_+, \quad 
\text{augment objective by } \sum_i \nu_i\,(u_i-\widehat{\phi}^{(i)}_t).
\]

---

## 8) Regularization (Global and Per-Agent)
\[
\mathcal{R}(\theta,\delta)=\frac{\lambda}{2}\|W\|_F^2+\frac{\lambda}{2}\|b\|_2^2+\mu_\delta\sum_i\|\delta^{(i)}\|_2^2\quad (\text{or }\mu_{\text{grp}}\sum_i\|\delta^{(i)}\|_2).
\]
Projection to centered subspace each round to enforce \(\sum_i \delta^{(i)}=0\).

---

## 9) Privacy and Accounting
Clip \(\|g\|_2\le C_g\) and release \(g+\mathcal{N}(0,\sigma^2 I)\). Over rounds with sampling rate \(q\), use an RDP/moments accountant to track cumulative \((\varepsilon,\delta)\). Choose \(\sigma\) to meet a target budget given \((T,q,C_g)\).

---

## 10) Off-Policy Evaluation (OPE)
Logged dataset \(\mathcal{D}=\{(x_i,a_i,r_i,p_i)\}_{i=1}^n\), target policy \(\pi_e\).

- **IPS (one-step bandit):**
\[
\widehat V_{\mathrm{IPS}}=\frac{1}{n}\sum_{i=1}^n \frac{\pi_e(a_i\mid x_i)}{\max(p_i,\epsilon)}\,r_i.
\]
- **SNIPS (self-normalized IPS):**
\[
w_i=\frac{\pi_e(a_i\mid x_i)}{\max(p_i,\epsilon)},\quad 
\widehat V_{\mathrm{SNIPS}}=\frac{\sum_i w_i r_i}{\sum_i w_i}.
\]
- **DR (doubly robust):**
\[
\widehat V_{\mathrm{DR}}=\frac{1}{n}\sum_{i=1}^n \Big[\hat r(x_i)+w_i\,(r_i-\hat r(x_i))\Big].
\]
- **WDR (weighted DR):**
\[
\widehat V_{\mathrm{WDR}}=\frac{\sum_i w_i\,\big[\hat r(x_i)+r_i-\hat r(x_i)\big]}{\sum_i w_i}
=\frac{\sum_i w_i r_i}{\sum_i w_i}\quad (\text{if }\hat r \text{ predicts under }\pi_e).
\]
- **SWITCH-DR:** use DR when \(w_i\le \tau\); otherwise plug \(\hat r(x_i)\).

**Uncertainty:** percentile bootstrap CIs or empirical-Bernstein bounds on IPS/SNIPS. Ensure \(\epsilon>0\) clipping.

---

## 11) Complete Loop (No Code)
For each round \(t\) and agent \(i\):  
(1) form \(X_t^{(i)}\); (2) sample \(a_t^{(i)}\sim\pi_\theta^{(i)}(\cdot\mid x_t^{(i)})\) (Section 1);  
(3) log \(p_t^{(i)}\); (4) observe \(r_t^{(i)}\) (Section 2); (5) store \((X,a,r,p)\);  
(6) periodically update \(W,b,\delta\) by IS/DR objective (Section 3) + regularizers (Section 8);  
(7) optional policy-gradient step using \(A_\psi=r-V_\psi\) (Section 4);  
(8) aggregate via centralized/FedAvg/decentralized with consistent formulas (Section 6);  
(9) update prices or load forecasts (Section 5); (10) enforce fairness with duals \(\nu\) (Section 7);  
(11) apply DP accounting (Section 9); (12) evaluate with OPE (Section 10).

---

## 12) Defaults
\(\tau_0{=}1,\ \tau_1{=}0.5,\ D_\tau{=}10^3,\ \epsilon_0{=}0.3,\ \epsilon_1{=}0.01,\ D{=}10^3,\ \lambda{=}10^{-6},\ \mu_\delta{=}10^{-3},\ \beta{=}0.2,\ \rho{=}0.05,\ \alpha{=}0.8,\ \gamma{=}0.01.\)

---

**Summary:** \(s_\theta\) is now explicitly defined for both value estimation and policy logits; \(\delta^{(i)}\) has regularization and centering; exploration includes Boltzmann schedules; prices/load forecasts are clarified; fairness uses explicit KPIs and dual updates; aggregation is consistent; rewards are modeled precisely; and OPE adds IPS/SNIPS/DR variants with uncertainty.
