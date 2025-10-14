# Fixed-Point Layers: A Math-Logic Assignment

This companion article develops the fixed-point ladders requested in the assignment.
We work systematically from foundations to optimisation shortcuts, documenting the
intermediate diagrams that help build intuition. Each section mirrors the prompts
from Parts A–I and keeps the notation consistent with the main README.

---

## Part A — Foundations (Existence)

### A1. Posets & Lattices

- A **chain** in a poset \((P,\leq)\) is a subset \(C\subseteq P\) such that any
  two elements are comparable: for all \(x,y\in C\), either \(x\leq y\) or
  \(y\leq x\).
- An **upper bound** of a subset \(S\subseteq P\) is an element \(u\in P\) with
  \(s\leq u\) for all \(s\in S\).
- A **least upper bound** (lub) or **supremum** of \(S\) is an upper bound
  \(u\) such that any other upper bound \(u'\) satisfies \(u\leq u'\).
- A poset is a **complete lattice** when every subset \(S\subseteq P\) has both
  a supremum \(\sup S\) and an infimum \(\inf S\).

**Proposition.** For any set \(S\), the powerset poset \((\mathcal{P}(S),\subseteq)\)
forms a complete lattice.

**Proof.** Given any family \(\{A_i\}_{i\in I}\subseteq\mathcal{P}(S)\), the union
\(\bigcup_{i\in I} A_i\) is an upper bound and the smallest such under inclusion,
so it is the supremum. Dually the intersection \(\bigcap_{i\in I} A_i\) is the
infimum. Both operations are well defined for arbitrary index sets, hence
\(\mathcal{P}(S)\) is a complete lattice. ∎

### A2. Monotone Maps & Fixpoints

A function \(T:P\to P\) on a poset is **monotone** when \(x\leq y\Rightarrow
T(x)\leq T(y)\).

**Knaster–Tarski Theorem.** If \(P\) is a complete lattice and \(T:P\to P\) is
monotone, then the set of fixed points of \(T\) is a complete lattice. In
particular the least and greatest fixed points \(\mathrm{lfp}(T)\) and
\(\mathrm{gfp}(T)\) exist.

**Proof.** Consider the set \(\mathcal{F}=\{x\in P : T(x)\leq x\}\) of
pre-fixpoints. Let \(p=\inf \mathcal{F}\). Because \(P\) is complete, \(p\)
exists. Monotonicity implies \(T(p)\leq T(x)\leq x\) for all \(x\in\mathcal{F}\),
so \(T(p)\leq p\). But \(T(p)\) is itself a pre-fixpoint, whence \(p\leq T(p)\).
Therefore \(T(p)=p\), proving \(p\) is the least fixed point. A symmetrical
argument applied to the set of post-fixpoints \(\{x : x\leq T(x)\}\) yields the
greatest fixed point. Any meet or join of fixed points remains a fixed point, so
the fixed points inherit a complete lattice structure. ∎

### A3. Kleene Iteration (Layers)

Assume \(P\) is a complete lattice with bottom element \(\bot\) and
\(T:P\to P\) is \(\omega\)-continuous (it preserves suprema of countable chains).
Define \(L_0=\bot\) and \(L_{n+1}=T(L_n)\).

**Claim.** \(\mathrm{lfp}(T)=\sup_{n<\omega} L_n\).

**Proof.** Monotonicity ensures \(L_0\leq L_1\leq\cdots\), so the supremum
\(L=\sup_{n<\omega} L_n\) exists. By \(\omega\)-continuity
\(T(L)=T(\sup_n L_n)=\sup_n T(L_n)=\sup_n L_{n+1}=L\), hence \(L\) is a
fixed point. Any pre-fixpoint \(x\) (with \(T(x)\leq x\)) bounds the chain from
above: \(L_n\leq x\) for all \(n\), so \(L\leq x\). Therefore \(L\) is the least
fixed point. ∎

The canonical ladder appears as:

```
L0 = ⊥  --T-->  L1  --T-->  L2  --T-->  ⋯
                                \
                                 \-- sup_n L_n = lfp(T)
```

---

## Part B — Concrete Reachability (Graph as Powerset Operator)

We study the graph \(G=(V,E)\) with
\(V=\{a,b,c,d,e\}\) and edges
\(E=\{(a,b),(a,c),(b,d),(c,d),(d,e)\}\). The operator
\(T:\mathcal{P}(V)\to\mathcal{P}(V)\) is
\(T(X)=\{a\}\cup \{v\in V : \exists u\in X,\ (u,v)\in E\}\).

### B1. Monotonicity and \(\omega\)-Continuity

If \(X\subseteq Y\) then every successor of a node in \(X\) is also a successor of
some node in \(Y\); the adjoined start node \(a\) is unaffected. Hence
\(T(X)\subseteq T(Y)\), showing monotonicity. Because \(T\) is defined via finite
unions over edges, it preserves suprema of \(\omega\)-chains: for
\(X_0\subseteq X_1\subseteq\dots\),
\(T(\bigcup_n X_n)=\{a\}\cup \bigcup_n \mathrm{Succ}(X_n)=\bigcup_n T(X_n)\).
Thus \(T\) is \(\omega\)-continuous.

### B2. Layers

Starting from \(L_0=\varnothing\):

| n | Layer \(L_n\)           |
|---|-------------------------|
| 0 | \(\varnothing\)          |
| 1 | \(\{a\}\)               |
| 2 | \(\{a,b,c\}\)           |
| 3 | \(\{a,b,c,d\}\)         |
| 4 | \(\{a,b,c,d,e\}\)       |
| 5 | \(\{a,b,c,d,e\}\) (stable) |

Diagrammatically:

```
∅ = L0  --T-->  {a}  --T-->  {a,b,c}  --T-->  {a,b,c,d}  --T-->  {a,b,c,d,e}
                                                              \
                                                               \-- stable lfp
```

### B3. Least Fixed Point and Prefixpoints

At \(n=4\) (and thereafter) the layer stabilises at
\(R=\{a,b,c,d,e\}\). Directly:
\(T(R)=\{a\}\cup\{b,c,d,e\}=R\), so \(R\) is a fixed point. If
\(X\subseteq V\) satisfies \(T(X)\subseteq X\), then \(a\in X\) and whenever a
node enters \(T(X)\) its predecessors must already lie in \(X\). By induction on
path length from \(a\), every vertex in \(R\) must belong to \(X\), so
\(R\subseteq X\). Hence \(R=\mathrm{lfp}(T)\).

### B4. "Best Layer" Selection

Costs: \(c(a)=3\), \(c(b)=1\), \(c(c)=2\), \(c(d)=5\), \(c(e)=2\).
Define \(C(L_n)=\sum_{v\in L_n} c(v)\) with \(C(L_0)=0\). The marginal gains
\(g_n=C(L_n)-C(L_{n-1})\) yield:

| n | \(C(L_n)\) | \(g_n\) |
|---|-------------|---------|
| 0 | 0           | –       |
| 1 | 3           | 3       |
| 2 | 6           | 3       |
| 3 | 11          | 5       |
| 4 | 13          | 2       |
| 5 | 13          | 0       |

The smallest \(n\) with \(g_n\leq 1\) is \(n^\star=5\), selecting
\(L_{n^\star}=\{a,b,c,d,e\}\). Stabilisation triggers the threshold: once the
layer stops expanding, diminishing returns reach zero.

---

## Part C — Logic via Least Fixpoints (µ-Style Semantics)

Let \(\mathrm{next}(S)=\{v : \exists u\in S,\ (u,v)\in E\}\).

### C1. Formula Evaluation

The formula \(\mu X.\big(\{a\}\cup \mathrm{next}(X)\big)\) corresponds exactly
to the operator \(T\) from Part B. Kleene iteration therefore reproduces the
layers \(L_n\) above, converging to \(\mathrm{lfp}(T)=\{a,b,c,d,e\}\).

### C2. Dual Greatest Fixpoint

Let \(\mathrm{prev}(S)=\{u : \exists v\in S,\ (u,v)\in E\}\). Consider
\(\nu Y.\big(V\setminus \mathrm{prev}(V\setminus Y)\big)\). Intuitively this
collects states whose outgoing edges remain inside the set: a node exits the set
iff it points to a node already excluded.

Starting the coinductive iteration at \(Y_0=V\), we obtain
\(V\setminus Y_0=\varnothing\), \(\mathrm{prev}(\varnothing)=\varnothing\), hence
\(Y_1=V\). The process stabilises immediately, so the greatest fixpoint is the
entire vertex set \(V\). All nodes in this DAG have successors either inside the
set or none at all, so none are forced out.

---

## Part D — Optimisation via Contraction (Shortcut with Guarantees)

### D1. Banach Fixed-Point Theorem

**Theorem.** Let \((X,\|\cdot\|)\) be a complete normed vector space and
\(T:X\to X\) a contraction: there exists \(0<\alpha<1\) with
\(\|T(x)-T(y)\|\leq \alpha\|x-y\|\) for all \(x,y\). Then:

1. \(T\) has a unique fixed point \(x^\star\in X\).
2. For any \(x_0\in X\) the sequence \(x_{n+1}=T(x_n)\) converges to \(x^\star\)
   with geometric rate: \(\|x_n-x^\star\|\leq \alpha^n\|x_0-x^\star\|\).

**Proof.** Define \(x_{n+1}=T(x_n)\). The contraction property yields
\(\|x_{n+1}-x_n\|\leq \alpha^n\|x_1-x_0\|\). The geometric series converges, so
\((x_n)\) is Cauchy and thus converges (completeness). Let the limit be
\(x^\star\). Continuity of \(T\) (inherited from the contraction inequality)
ensures \(T(x^\star)=x^\star\). If \(y\) were another fixed point, then
\(\|x^\star-y\|=\|T(x^\star)-T(y)\|\leq \alpha\|x^\star-y\|\). With
\(0<\alpha<1\) this forces \(x^\star=y\). The inequality for \(\|x_n-x^\star\|
\) follows by induction: \(\|x_{n+1}-x^\star\|=\|T(x_n)-T(x^\star)\|
\leq\alpha\|x_n-x^\star\|\). ∎

### D2. A 4-Dimensional Toy Model

Let \(r\in[0,1]^4\) (use the numerical vector from the README) and
\(0<\lambda<1\). For a matrix \(A\) with spectral norm \(\|A\|\leq\beta<1\), define
\(T(x)=\lambda r + (1-\lambda) A x\).

1. **Contraction and Fixpoint.** The map is affine with linear part
   \((1-\lambda)A\). Any matrix norm subordinate to \(\|\cdot\|\) satisfies
   \(\|T(x)-T(y)\|\leq (1-\lambda)\beta\,\|x-y\|\). Because \((1-\lambda)\beta<1\),
   Banach applies with contraction constant \(\alpha=(1-\lambda)\beta\). Solving
   \(x=\lambda r + (1-\lambda)A x\) yields
   \((I-(1-\lambda)A)x=\lambda r\), hence
   \(x^\star=(I-(1-\lambda)A)^{-1}\lambda r\).
2. **Error Bound.** Starting from arbitrary \(x_0\), the Banach estimate gives
   \(\|x_n-x^\star\|\leq \alpha^n\|x_0-x^\star\|\).
3. **Concrete Choice.** Take \(\lambda=0.4\) and
   \(A=\mathrm{diag}(0.2,0.3,0.25,0.1)\). Then \(\beta=0.3\) and the contraction
   factor is \(\alpha=(1-0.4)\times 0.3=0.18\). Using the README vector
   \(r=[0.27132374,0.29026431,0.075,0.36341195]^\top\),
   \(x^\star\approx[0.12333,\ 0.14159,\ 0.03529,\ 0.15464]^\top\).

Shortcut diagram:

```
x0 --T--> x1 --T--> x2 --T--> ··· (errors shrink by ×0.18 each step)
 \
  \-- Banach guarantee jumps straight to x★
```

---

## Part E — Closure-Based Stabilisation (Finite-Time Shortcut)

### E1. Sound Acceleration

Let \(c:P\to P\) be a closure operator (monotone, extensive, idempotent) and
consider \(L^{c}_{0}=\bot\), \(L^{c}_{n+1}=c(T(L^{c}_{n}))\).

1. **Dominance.** Prove by induction:
   \(L_0\leq L^{c}_{0}\). If \(L_n\leq L^{c}_{n}\), then
   \(L_{n+1}=T(L_n)\leq T(L^{c}_{n})\leq c(T(L^{c}_{n}))=L^{c}_{n+1}\).
2. **Suprema.** The dominance implies \(\sup_n L_n\leq\sup_n L^{c}_{n}\).
3. **Fixpoint Ordering.** Assume \(c\circ T\leq T\circ c\). Any fixed point
   \(x\) of \(T\) above \(\bot\) satisfies
   \(c(T(x))\leq T(c(x))\). But \(T(x)=x\), and extensivity gives
   \(x\leq c(x)\). Therefore \(c(T(x))=c(x)\leq T(c(x))\), showing \(c(x)\) is a
   post-fixpoint of \(T\). Consequently
   \(\mathrm{lfp}(T)\leq \mathrm{lfp}(c\circ T)\); acceleration never undershoots
   the true least fixed point.

### E2. Finite Convergence Under ACC

If \(P\) satisfies the ascending chain condition (ACC) and the image of \(c\) has
no infinite strictly ascending chains, the accelerated sequence \(L^{c}_n\)
stabilises after finitely many steps: once the iteration enters \(\mathrm{Im}(c)\)
it can only move finitely often before reaching a fixpoint inside the finite
poset.

**Application to Part B.** Define \(c(X)=X\cup \{v\in V : \mathrm{dist}(a,v)\leq 2\}\).
This closure adds the "two-hop" neighbourhood around the start node regardless of
\(X\). Applying \(c\) once returns \(\{a,b,c,d\}\), and further applications are
idempotent. The accelerated ladder becomes:

```
L0^c = ∅
L1^c = {a,b,c,d}
L2^c = {a,b,c,d}
```

Thus \(\mathrm{lfp}(c\circ T)=\{a,b,c,d\}\), an over-approximation of the true
reachability set \(\{a,b,c,d,e\}\). The closure trades precision for a two-step
certificate.

---

## Part F — "Best Layer" as an Optimisation Problem

Let \((L_n)_{n\geq 0}\) be a nondecreasing chain with utility \(U\) and cost
\(C\), both nonnegative. Define \(\Phi(n)=U(L_n)-\lambda C(L_n)\) for \(\lambda>0\).

### F1. Greedy 1/2-Approximation

Assume \(U\) is submodular (diminishing returns) and \(C\) is modular (additive).
The marginal gain satisfies
\(\Phi(n+1)-\Phi(n) = \big(U(L_{n+1})-U(L_n)\big) - \lambda\big(C(L_{n+1})-C(L_n)\big)\).
Submodularity ensures the marginal utility along the chain is nonincreasing,
while the marginal cost is fixed by modularity. The classic proof for greedy
submodular maximisation over a chain applies: once the marginal drops nonpositive,
continuing cannot increase the objective by more than the value already accrued,
so the greedy stopping point achieves at least half of the optimal \(\Phi\).∎

### F2. Execution on Part B

Use \(U(L_n)=|L_n|\) (node coverage) with the costs from B4 and \(\lambda=0.3\).
Computed values:

| n | \(|L_n|\) | \(C(L_n)\) | \(\Phi(n)\) |
|---|-----------|-------------|------------|
| 0 | 0         | 0           | 0.00       |
| 1 | 1         | 3           | 0.10       |
| 2 | 3         | 6           | 1.20       |
| 3 | 4         | 11          | 0.70       |
| 4 | 5         | 13          | 1.10       |
| 5 | 5         | 13          | 1.10       |

The greedy rule stops when \(\Phi(n+1)-\Phi(n)\leq 0\), which occurs between
\(n=2\) and \(n=3\). Therefore \(n^\star=2\) and \(L_{n^\star}=\{a,b,c\}\). This
matches the true maximiser of \(\Phi\) for the chosen \(\lambda\).

---

## Part G — DFS vs BFS as Different Operators

Work on \(\mathcal{P}(E)\), the lattice of edge sets ordered by inclusion.

### G1. Monotonicity

- \(T_{\text{BFS}}\) adds every outgoing edge from the vertices discovered at the
  current breadth layer. Enlarging the input edge set can only increase the pool
  of frontier vertices, so the output edges are monotone in the input.
- \(T_{\text{DFS}}\) reveals at most one new outgoing edge from the latest vertex
  on the search stack. Expanding the input set preserves or increases the stack,
  never removing edges, so the operator is also monotone.

### G2. Fixed Points and Layer Structure

Running \(T_{\text{BFS}}\) from \(\varnothing\) yields successive layers of edges
according to breadth-first depth, stabilising once all reachable edges
\(E_R=\{(a,b),(a,c),(b,d),(c,d),(d,e)\}\) are included. This is the least fixed
point and matches the minimal number of layers required to expose every edge.

For deterministic DFS (choose the stack order \(b\) before \(c\)), the ladder is:

```
∅ → {(a,b)} → {(a,b),(b,d)} → {(a,b),(b,d),(d,e)} → {(a,b),(b,d),(d,e),(a,c)} → E_R
```

The DFS operator still converges to \(E_R\), but the layer order differs and can
lag behind the BFS breadth expansion in the lattice order (some edges such as
\((a,c)\) appear late).

### G3. Closure Shortcut

Define \(c\) that maps any DFS frontier to the set of all edges discovered by a
breadth-first pass up to the same depth. Formally, given a DFS edge set, compute
its induced vertex depths and add all outgoing edges from vertices at those
depths. The composite \(c\circ T_{\text{DFS}}\) collapses each DFS layer onto the
corresponding BFS layer, matching the BFS convergence speed while retaining DFS's
stack discipline within a layer.

---

## Part H — Mini-Capstone: Modelling a "Current Stage"

Let \([0,1]^4\) carry the coordinatewise order. Define
\(T(x)=\min(\mathbf{1},\ r + Bx)\) with min applied elementwise, \(r\in[0,1]^4\)
and \(B\geq 0\) with row sums \(\leq 1\).

### H1. Monotonicity and Least Fixpoint

If \(x\leq y\) then \(Bx\leq By\) (all entries of \(B\) are nonnegative).
Adding \(r\) preserves order, and componentwise minimum with \(\mathbf{1}\)
is also monotone. Thus \(T\) is monotone on the complete lattice \([0,1]^4\);
Knaster–Tarski guarantees a least fixed point.

### H2. Layers

Symbolically the iteration reads
\(x^{(0)}=\mathbf{0}\), \(x^{(n+1)}=\min(\mathbf{1},\ r + Bx^{(n)})\).
For a concrete choice \(B=\mathrm{diag}(0.25,0.30,0.20,0.25)\) we obtain:

| n | \(x^{(n)}\)                                  |
|---|----------------------------------------------|
| 0 | \((0, 0, 0, 0)\)                            |
| 1 | \((0.2713, 0.2903, 0.0750, 0.3634)\)        |
| 2 | \((0.3392, 0.3773, 0.0900, 0.4543)\)        |
| 3 | \((0.3561, 0.4035, 0.0930, 0.4770)\)        |
| 4 | \((0.3604, 0.4113, 0.0936, 0.4827)\)        |
| 5 | \((0.3614, 0.4137, 0.0937, 0.4841)\)        |

The chain ascends monotonically toward the least fixed point (approximately the
limit listed in row 5).

### H3. Shortcuts

1. **Contraction via \(\|\cdot\|_\infty\).** If \(\|B\|_\infty<1\) (maximum row sum
   strictly less than 1), the affine map \(S(x)=r+Bx\) is a contraction in
   \(\|\cdot\|_\infty\). Banach yields a unique fixed point \(x^\dagger\). Because
   \(S(x)\leq T(x)\leq \mathbf{1}\), we have \(x^\dagger\leq \mathrm{lfp}(T)\leq
   \mathbf{1}\). The contraction shortcut bounds the true least fixed point from
   below.
2. **Quantised Closure.** For \(0<\delta\leq 1\), define
   \(c(x)=\lceil x\rceil_\delta\): each coordinate is rounded up to the nearest
   multiple of \(\delta\) (clipped at 1). This map is monotone, extensive, and
   idempotent. Iterating \(c\circ T\) therefore stabilises in at most
   \(1/\delta\) steps per coordinate because only finitely many multiples occur.
   Choosing utility as the average of the four coordinates, one can run the
   accelerated ladder and apply the Part F rule to pick the preferred stopping
   layer with bounded exploration.

Ladder sketch:

```
(0,0,0,0) → T → … → lfp(T)
 |                  \
 |                   \-- quantised shortcut via c
 \-- contraction shortcut via S
```

---

## Part I — Reflection

Layers via Kleene iteration, optimisation over chains, and shortcuts through
contraction or closure form a unified recipe. We begin with a monotone operator
on a complete lattice and generate a ladder of approximants from the bottom.
Submodular or cost-aware criteria let us pick a practical stopping layer before
reaching the formal fixed point, interpreting each layer as a "best effort"
approximation aligned with our utility. When convergence needs guarantees, we
introduce shortcuts: Banach contractions promise geometric decay, while closure
operators provide finite acceleration with sound over-approximations.

These techniques complement each other. Kleene iteration exposes the qualitative
structure of the solution, optimisation criteria highlight when additional layers
add little value, and shortcuts certify that we can either stop early with a
provable bound or safely jump ahead. Together they transform abstract fixed-point
problems into actionable workflows for reachability, logical semantics, or
self-calibrating models.

Favourite ladder (Part B) with selections and shortcuts:

```
⊥ → {a} → {a,b,c} → {a,b,c,d} → {a,b,c,d,e} = lfp(T)
|                    \
|                     \-- greedy stop at {a,b,c}
\-- closure shortcut to {a,b,c,d}
```
