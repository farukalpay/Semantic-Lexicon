# Semantic Lexicon

[![CLI Tests](https://github.com/farukalpay/Semantic-Lexicon/actions/workflows/cli-tests.yml/badge.svg)](https://github.com/semantic-lexicon/Semantic-Lexicon/actions/workflows/cli-tests.yml)

Semantic Lexicon is a NumPy-first research toolkit that demonstrates persona-aware semantic modelling. The project packages a compact neural stack consisting of intent understanding, a light-weight knowledge network, persona management, and text generation into an automated Python library and CLI.

The name reflects the long-standing academic concept of the [semantic lexicon](https://en.wikipedia.org/wiki/Semantic_lexicon); this repository contributes an applied, open implementation that operationalises those ideas for persona-aware experimentation.

---

## Contents

- [Quick Start](#quick-start)
- [Citation](#citation)
- [Features](#features)
- [Installation](#installation)
- [Project Layout](#project-layout)
- [CLI Walkthrough](#cli-walkthrough)
- [Streams & Clipboard](#streams--clipboard)
- [TADKit](#tadkit--drop-in-logits-processor-and-cli)
- [PersonaRAG](#personarag--exp3-personas-with-decode-time-truth-gates)
- [Development Workflow](#development-workflow)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Author's Note](#authors-note)
- [Contact & Legal](#contact--legal)

---

## Quick Start

```bash
# 1. Install the CLI and core library
pip install "semantic-lexicon @ git+https://github.com/farukalpay/Semantic-Lexicon.git"

# 2. Materialise the sample workspace
semantic-lexicon prepare \
  --intent src/semantic_lexicon/data/intent.jsonl \
  --knowledge src/semantic_lexicon/data/knowledge.jsonl \
  --workspace artifacts

# 3. Train and generate
semantic-lexicon train --workspace artifacts
semantic-lexicon generate "Outline transformer basics" --workspace artifacts
```

Sample output:

```text
Persona: generic
Response:
1. Transformers stack self-attention blocks so each token can weigh every other token without recurrent passes.
2. Positional encodings add order-aware vectors that let attention keep track of word positions.
3. Feed-forward layers plus residual and layer-norm steps stabilise optimisation and let depth scale cleanly.
```

- Prefer `semantic-lexicon generate -` to pipe prompts from other tools (`echo question | semantic-lexicon generate -`).
- `semantic-lexicon clipboard --workspace artifacts` mirrors the same workflow but seeds the prompt from your system clipboard.

## Citation

Semantic Lexicon operationalises the reproducible persona-aware pipeline introduced in the accompanying preprint. If you build on this toolkit, please cite the work so other researchers can trace the connection between the paper's methodology and this implementation.

```bibtex
@misc{alpay2025reproduciblescalablepipelinesynthesizing,
      title={A Reproducible, Scalable Pipeline for Synthesizing Autoregressive Model Literature}, 
      author={Faruk Alpay and Bugra Kilictas and Hamdi Alakkad},
      year={2025},
      eprint={2508.04612},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.04612}, 
}
```

You can read the preprint online at [https://arxiv.org/abs/2508.04612](https://arxiv.org/abs/2508.04612); it documents the scalable data curation and evaluation strategy that directly powers the automation, diagnostics, and persona controls exposed by this repository.

## Features

**Core modelling**
- Modular architecture spanning embeddings, intents, knowledge graphs, personas, and persona-aware generation.
- Deterministic NumPy training loops delivering reproducible optimisation.
- Graph-driven knowledge curation using SPPMI weighting, smoothed relevance, and greedy facility-location selection.

**Automation & tooling**
- Typer-powered CLI for preparation, training, diagnostics, and generation.
- Extensible configuration with dataclass-backed YAML/JSON loading.
- Diagnostics covering embeddings, intents, knowledge neighbours, personas, and generation previews.
- Documentation plus regression safeguards via MkDocs, pytest, prompt evaluations, ruff, mypy, and black.

**Decision-making & safety**
- EXP3 persona selection utilities for adversarial style experiments.
- Analytical guarantees through composite reward shaping, calibration, and regret tooling.
- Primal–dual safety tuning that balances exploration, pricing, and knowledge gates until residuals vanish.

## Installation

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Semantic Lexicon** using pip's `name @ URL` syntax (avoids future deprecation warnings while matching the hosted repository):

   ```bash
   pip install "semantic-lexicon[dev,docs] @ git+https://github.com/farukalpay/Semantic-Lexicon.git"
   ```

### Optional extras

Use the table below to tailor optional extras to your workflow:

| Goal | Command |
| --- | --- |
| Minimal CLI & library | `pip install "semantic-lexicon @ git+https://github.com/farukalpay/Semantic-Lexicon.git"` |
| Docs + developer tooling | `pip install "semantic-lexicon[dev,docs] @ git+https://github.com/farukalpay/Semantic-Lexicon.git"` |
| TADKit demos (needs PyTorch + Streamlit) | `pip install "semantic-lexicon[tadkit] @ git+https://github.com/farukalpay/Semantic-Lexicon.git"` |
| PersonaRAG demos (LangChain/LangGraph stack) | `pip install "semantic-lexicon[personarag] @ git+https://github.com/farukalpay/Semantic-Lexicon.git"` |

> **Note:** `tadkit` relies on PyTorch for logits processing. Install a CPU build with
> `pip install torch --index-url https://download.pytorch.org/whl/cpu` if it is not already
> included in your environment.

## Project Layout

```
src/semantic_lexicon/
├── cli.py                # Typer CLI entry point
├── config.py             # Dataclass-backed configuration helpers
├── embeddings.py         # GloVe-style embeddings and persistence
├── intent.py             # NumPy multinomial logistic regression intents
├── knowledge.py          # Simple relation network with gradient updates
├── persona.py            # Persona profiles and blending logic
├── generator.py          # Persona-aware response generation
├── model.py              # Orchestration façade
├── training.py           # Training pipeline and diagnostics integration
├── diagnostics.py        # Structured diagnostics reports
├── utils/                # Tokenisation, seeding, and I/O helpers
└── data/                 # Sample intent & knowledge datasets for tests
```

> **Tip:** `examples/` mirrors these modules with runnable notebooks and scripts so you can jump from the reference implementation to experiments quickly.

## CLI Walkthrough

Follow the full pipeline when you want to rebuild artefacts from raw data.

1. **Prepare the corpus** (optional if using bundled sample data):

   ```bash
   semantic-lexicon prepare --intent src/semantic_lexicon/data/intent.jsonl --knowledge src/semantic_lexicon/data/knowledge.jsonl --workspace artifacts
   ```

2. **Train the model** (uses processed datasets in `artifacts/`):

   ```bash
   semantic-lexicon train --workspace artifacts
   ```

The CLI saves embeddings, intent weights, and knowledge matrices to the workspace directory.

3. **Run diagnostics**:

   ```bash
   semantic-lexicon diagnostics --workspace artifacts --output diagnostics.json
   ```

   The command prints a JSON summary to stdout and optionally writes the report to disk.

   ```json
   {
     "embedding_stats": {"vocab_size": 512, "mean_norm": 8.42, "std_norm": 0.77},
     "intent_accuracy": 0.94,
     "knowledge_graph": {"edges": 2048, "avg_degree": 3.1},
     "personas_evaluated": ["generic", "tutor", "coach"]
   }
   ```

4. **Generate responses**:

   ```bash
   semantic-lexicon generate "Explain neural networks" --workspace artifacts --persona tutor
   ```

   Example response:

   ```text
   Persona: tutor
   Response:
   - Neural networks stack layers of weighted sums and nonlinear activations that learn feature detectors automatically.
   - Training adjusts the weights with backpropagation so the model minimises prediction error on labelled examples.
   - Regularisation (dropout, weight decay) keeps the learned representations from overfitting and improves generalisation.
   ```

Optional CLI calls:

- Tight briefing

  ```bash
  semantic-lexicon ask-tight "How can I improve my research talks?" --workspace artifacts --bullets 3
  ```

  ```text
  • Outline the three-part story: problem framing, insight, and next steps.
  • Rehearse with a timer so every visual lands within the planned beats.
  • Script a final call-to-action that leaves the audience with one clear task.
  ```

- Inspect knowledge selection

  ```bash
  semantic-lexicon knowledge "Summarise convolutional neural networks" --workspace artifacts
  ```

  ```json
  {
    "concepts": [
      "convolutional kernels capture spatial structure",
      "pooling layers balance invariance with detail",
      "feature maps highlight class-specific patterns"
    ],
    "relevance": 0.87,
    "coverage": 0.74,
    "diversity": 0.69
  }
  ```

## Truth-Aware Decoding Walkthrough

Truth-aware decoding (TAD) combines a model's logits with declarative knowledge supplied by one or more safety oracles. Each decode step (a) queries the model for logits, (b) consults the oracle for a boolean allow/block mask plus diagnostic labels, (c) computes the probability mass that remains safe, and (d) either selects the highest-probability safe token or abstains when the safe mass falls below a configurable threshold. The [`semantic_lexicon.decoding_tad.truth_aware_decode`](src/semantic_lexicon/decoding_tad.py) loop is pure NumPy and logs every decision in a `TADStepLog`, making it easy to audit or integrate into research pipelines.

### Reproducing the toy capital-of demo

The repository includes a fully-worked, research-grade example that resolves the prompt *“Paris is the capital of …”* against a knowledge-base oracle. The script reproduces the toy model from the TAD unit tests, injects a fact table with a single triple `(Paris, capital_of, France)`, and records all decode-time telemetry to disk.

```bash
python examples/truth_aware_decode_demo.py
```

Running the demo prints the safe decoding trace and saves the structured log to `examples/logs/paris_capital_truth_aware_decode.json`:

```
Prompt tokens: <BOS> Paris is the capital of
Generated tokens: France <EOS>
Log written to examples/logs/paris_capital_truth_aware_decode.json
```

The JSON log captures every metric needed for a forensic audit. Each entry in `steps` reports the decode index `t`, the safe probability mass `pi_safe` (after masking), the selected token, the number of blocked vocabulary entries, and the oracle-provided reason labels. A shortened excerpt is shown below; the full file is part of the repository so you can cite or diff it in papers and lab notebooks.

```json
{
  "prompt_tokens": ["<BOS>", "Paris", "is", "the", "capital", "of"],
  "generated_tokens": ["France", "<EOS>"],
  "abstained": false,
  "steps": [
    {
      "t": 0,
      "pi_safe": 0.3390092760113778,
      "picked_token": "France",
      "blocked_count": 3,
      "reasons_for_picked": ["kb:required_object"]
    },
    {
      "t": 1,
      "pi_safe": 1.0,
      "picked_token": "<EOS>",
      "blocked_count": 0,
      "reasons_for_picked": []
    }
  ]
}
```

From here you can: (1) swap in the graph-backed oracle to ground against a larger knowledge base, (2) set `TADConfig.abstain_token` to emit a sentinel when `pi_safe` drops below the threshold, and (3) feed the logged `pi_safe` sequence into your own reliability analyses (e.g., cumulative risk bounds or safe-mass histograms). Because `truth_aware_decode` works with any `Oracle` implementation, PhD students can plug in bespoke symbolic checkers—factuality verifiers, contradiction detectors, or mathematical solvers—without touching the decoding loop itself.

## TADKit — drop-in logits processor and CLI

The repository now exposes a standalone [`tadkit`](src/tadkit) package so you can pip-install the truth-aware decoding utilities outside of the monolithic CLI. TADKit mirrors the “expected product” shown in the product brief:

- `TruthOracle` turns CSV/JSON/YAML rules into prompt-activated constraints.
- `TADLogitsProcessor` plugs into `transformers` generation loops and injects abstain tokens when a rule is violated.
- `TADTrace` logs token-level actions and renders console tables or Pandas dataframes for audits.
- `tadkit compile` converts spreadsheets to JSON payloads; `tadkit demo` spins up a tiny `sshleifer/tiny-gpt2` demo using the compiled oracle.
- `examples/tadkit_quickstart.py` and `examples/tadkit_streamlit_app.py` are copy-pasteable quickstarts, matching the walkthrough in the brief.

Install extras as needed:

```bash
pip install "git+https://github.com/farukalpay/Semantic-Lexicon.git#egg=semantic-lexicon[tadkit]"
tadkit compile capitals.csv --out oracle.json --tokenizer gpt2
tadkit demo --oracle oracle.json --model sshleifer/tiny-gpt2 \
  --prompt "Q: What is the capital of France?\nA:"
```

> **Note:** `tadkit` relies on PyTorch for logits processing. Install a CPU
> build with `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> if you do not already have `torch` available.

## PersonaRAG — EXP3 personas with decode-time truth gates

PersonaRAG is a thin layer on top of LangChain/LangGraph that routes tone, enforces truth, and records feedback telemetry. The [`personarag`](src/personarag) package exposes:

- `BrandStyle` persona descriptors.
- `PersonaPolicyEXP3` — contextual EXP3 with weight telemetry and bulk feedback helpers.
- `KnowledgeGate` — wraps LangChain LLMs and installs `TADLogitsProcessor` when the underlying model exposes Hugging Face hooks.
- `examples/personarag_quickstart.py` — the complete “expected product” script from the brief.

Install with optional dependencies when you want the full LangChain stack:

```bash
pip install "git+https://github.com/farukalpay/Semantic-Lexicon.git#egg=semantic-lexicon[personarag]"
python examples/personarag_quickstart.py
```

Decode-time gating is enabled automatically for Hugging Face models (local or
via LangChain wrappers). Hosted chat models (e.g., OpenAI) receive trace
metadata only.

`KnowledgeGate` attaches `trace.events` to `response_metadata` (when available), so observability dashboards can render trace heatmaps alongside persona win-rates and abstain telemetry.

## Knowledge Selection Playbook

The knowledge selector now treats every AGENTS.md instruction as a hard feasibility constraint. Broad concepts can still join the shortlist, but only when they collaborate with prompt-relevant anchors *and* all group bounds are respected.

> **Note:** The full mathematical specification for the selector — including the object definitions, scoring components, constraints, and optimisation guarantees — now lives in [`docs/articles/knowledge-selector.tex`](docs/articles/knowledge-selector.tex). The README keeps the practitioner-focused workflow and validation guidance below; consult the article whenever you need the derivations or precise notation.

### Workflow

1. **Graph construction.** Estimate shifted PPMI weights with smoothing \(p(i)^\gamma\); derive \(S\), \(D\), \(L\), and \(P\).
2. **Relevance smoothing.** Compute raw cosine relevance, solve the graph-regularised system, and classify on/off-topic nodes via the topic threshold.
3. **Anchoring.** Select anchors, compute personalised PageRank bridges, and form soft gates \(g_i\).
4. **Group configuration.** Register AGENTS.md groups with `set_concept_groups` and interval bounds with `set_group_bounds`; the selector automatically adds on/off-topic ratios.
5. **Greedy selection.** Evaluate admissible candidates, compute marginal coverage, cohesion, collaboration, and diversity, and add the best concept while updating group capacities.
6. **Reporting.** Emit the chosen concepts plus relevance, coverage, cohesion, collaboration, diversity, raw knowledge score, and mean gate.

Defaults \((\alpha, \lambda, \mu, \gamma, \tau, \lambda_1, \lambda_2, K, \tau_g, \text{on/off ratios}) = (0.12, 0.08, 0.5, 0.35, 0.1, 0.6, 0.4, 12, 0.08, 0.6/0.2/0.4)) ship in `KnowledgeConfig`. Additional per-group intervals can be supplied at runtime. The legacy phrase planner (MMR phrase selection with PMI bonuses) remains available inside the generator for reproducibility.
Use the CLI to inspect the concepts chosen for a prompt without rendering a full response:

```bash
semantic-lexicon knowledge "Explain matrix multiplication" --workspace artifacts
```

The JSON payload now includes gated relevance, coverage, cohesion, collaboration reward, log-det diversity, the raw knowledge score,
and the mean gate value across selected concepts.

### Go/No-Go Validation

Before shipping a new persona or pricing configuration, run the Go/No-Go suite to certify that knowledge selection obeys AGENTS.md, the deployment policy respects the exploration rules, and the off-policy lift is trustworthy.

1. **Rule feasibility.** Map each concept to its groups and bounds, count how many selections fall inside every group, and reject whenever any lower or upper bound is violated. `SelectionSpec` now bundles a `KnowledgeSignals` payload so the same object carries the calibrated knowledge metrics required later in the gate.

2. **Policy consistency.** For each logged step, rebuild the policy that was deployed using the stored logits, temperature, exploration mixture, and whichever penalty mode (prices or congestion) was active. The policy gate fails if any logged action falls below its exploration floor, if prices and congestion penalties are mixed, if the knowledge weight leaves the [0,1] range, or if the SNIPS floor dips below the exploration limit — guarding the AGENTS exploration guarantees.

3. **Off-policy value & fairness.** Using tuples (x_i, a_i, r_i, p_i) and the reconstructed target policy, compute SNIPS weights, the estimated value, and the effective sample size. Enforce a non-negative lower confidence bound on the lift, require the effective sample size to exceed one percent of the log length, and evaluate fairness either on action frequencies or KPI gaps via `FairnessConfig`.

4. **Price/congestion stability.** Aggregate the penalty vector each timestep and ensure the most recent window keeps total variation below the configured threshold. `StabilityCheckResult` records the peak deviation so you can tighten rho or beta when oscillations appear.

5. **Knowledge lift.** Compare the calibrated score and graph metrics captured in `KnowledgeSignals`. The gate demands the calibrated knowledge score stay above the trailing median and both coverage and cohesion deltas remain non-negative against the baseline selection size.

6. **Go/No-Go decision.** `run_go_no_go` wires the six checks together and emits a `GoNoGoResult` containing the selection feasibility, policy mode, OPE summary (with ESS target), stability diagnostics, and knowledge lift verdict. The `accepted` flag only flips to `True` when **every** gate passes. If any condition fails, follow the fix-once cascade in the specification — tweak the single knob (e.g., adjust `l_off`, `tau_g`, `eta`, or `rho`) and re-run the optimisation exactly once before re-testing.


### Primal–Dual Safety Gate Autotuning

Manual gate sweeps are still supported, but the preferred workflow is to run the projected primal–dual controller introduced in `semantic_lexicon.safety`. The controller now minimises the supplied objective while enforcing convex constraints, matching the textbook projected primal–dual loop.

```python
from semantic_lexicon.safety import (
    ConstraintSpec,
    GateBounds,
    ObjectiveSpec,
    run_primal_dual_autotune,
)

objective = ObjectiveSpec(
    function=lambda params: params["x1"] ** 2
    + params["x2"] ** 2
    - params["x1"]
    - params["x2"],
    gradient=lambda params: {
        "x1": 2.0 * params["x1"] - 1.0,
        "x2": 2.0 * params["x2"] - 1.0,
    },
)

constraints = [
    ConstraintSpec(
        "linear",
        lambda params: params["x1"] + params["x2"] - 1.0,
        gradient=lambda params: {"x1": 1.0, "x2": 1.0},
    )
]

result = run_primal_dual_autotune(
    objective,
    constraints,
    initial_parameters={"x1": 0.2, "x2": 0.8},
    parameter_names=("x1", "x2"),
    bounds={
        "x1": GateBounds(lower=0.0, upper=1.0),
        "x2": GateBounds(lower=0.0, upper=1.0),
    },
    primal_step=0.2,
    dual_step=0.4,
)

print("before", result.history[0])
print("after", result.parameters)
```

The first history entry captures the primal iterate after the initial step alongside its constraint violation, while the final snapshot records the tuned solution and dual multiplier. Swapping in exploration, fairness, or stability constraints follows the same pattern—only the callbacks change.

### Single-change presentation planner

When time only allows one tweak before a repeat talk, call `build_single_adjustment_plan()` to fetch a rehearsable experiment and a set of intent-hidden contingency moves. The helper keeps pacing and visuals frozen, picks *story beats* as the highest-leverage lever, and returns:

- A 20-minute rehearsal script that remaps the 12-minute slot into five beats, captures the headline you expect listeners to write down in each block, logs energy scores, and enforces a pass/fail line that demands fresh takeaways past minute seven.
- Five backup drills covering energy checkpoints, a slide trim for mixed audiences, a Q&A guardrail, a warmth-restoring micro-story, and a lighting plus breathing tweak for filler-word control.

```python
from semantic_lexicon.presentation import build_single_adjustment_plan

experiment, backups = build_single_adjustment_plan()
print(experiment.focus)
for move in backups:
    print(move.label)
```

Backups remain intent-hidden so you can pivot mid-practice without exposing the heuristic to the audience.

## Lightweight Q&A Demo

Semantic Lexicon can answer short questions after its bundled model components are trained. The stack is intentionally tiny, so
the phrasing is concise, but the generator now runs a compact optimisation loop that:

1. **Classifies intent** with the logistic-regression intent model.
2. **Builds noun-phrase and collocation candidates** whose adjacent tokens clear an adaptive pointwise mutual information (PMI)
   threshold, keeping multi-word ideas intact.
3. **Scores each candidate** via cosine relevance to the blended persona/prompt embedding, tf–idf salience, and a capped PMI
   cohesion bonus.
4. **Selects diverse topics** with Maximum Marginal Relevance (MMR) plus an n-gram overlap penalty so the guidance does not echo
   the question verbatim.
5. **Optimises knowledge coverage** by running the gated SPPMI graph objective (smoothed relevance, anchor gating, collaboration
   reward, log-det diversity, and group-aware constraints) and appending the resulting knowledge focus and related concepts.
6. **Aligns journaling actions** with the detected intent so each topic carries a concise Explore/Practice/Reflect-style cue.

1. Install the project in editable mode:

   ```bash
   pip install -e .
   ```

2. Run a quick script that trains the miniature model and generates answers for a few prompts:

   ```bash
   python - <<'PY'
   from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
   from semantic_lexicon.training import Trainer, TrainerConfig

   config = SemanticModelConfig()
   model = NeuralSemanticModel(config)
   trainer = Trainer(model, TrainerConfig())
   trainer.train()

   for prompt in [
       "How do I improve my public speaking?",
       "Explain matrix multiplication",
       "What is machine learning?",
       "Tips for staying productive while studying",
       "Clarify the concept of photosynthesis",
       "How can I organize my research presentation effectively?",
       "Define gravitational potential energy",
   ]:
       response = model.generate(prompt, persona="tutor")
       print(
           f"Prompt: {prompt}\\nResponse: {response.response}\\nKnowledge: {response.knowledge_hits}\\n"
       )
   PY
   ```

   Sample output after training the bundled data:

   ```text
   Prompt: How do I improve my public speaking?
   Persona: tutor
   Guidance:
   - Schedule deliberate practice sessions (record short talks, review pacing and emphasis).
   - Build a feedback loop with trusted listeners after each rehearsal.
   - Reflect on audience energy so you can adjust tone and gesture.
   Knowledge focus: practise short talks on camera.
   Related concepts: collect feedback from trusted listeners; rehearse openings and transitions; track energy cues across slides.

   Prompt: Explain matrix multiplication
   Persona: tutor
   Guidance:
   - Describe matrix multiplication as repeated dot products between rows and columns.
   - Connect the operation to linear transformations that reshape vectors.
   - Compare 2×2 and 3×3 cases to build intuition about scaling and rotation.
   Knowledge focus: review the row-by-column rule.
   Related concepts: connect matrix products to linear transformations; practise multiplying 2×2 and 3×3 matrices; interpret column-space changes.

   Prompt: Define gravitational potential energy
   Persona: tutor
   Guidance:
   - State that gravitational potential energy equals mass × gravity × height relative to a reference.
   - Show how choosing different reference frames shifts absolute values but not energy differences.
   - Link the concept to conservation of mechanical energy in simple motion problems.
   Knowledge focus: relate height changes to energy storage.
   Related concepts: draw free-body diagrams for objects at different heights; compare gravitational and elastic potential energy; highlight conservation across motion phases.
   ```

  These concise replies highlight the intentionally compact nature of the library's neural components—the toolkit is designed for
  research experiments and diagnostics rather than fluent conversation, yet it showcases how questions can be routed through the
  persona-aware pipeline.

  Running `python examples/quickstart.py` (or `PYTHONPATH=src python examples/quickstart.py` from a checkout) produces a combined
  generation preview and the new intent-selection walkthrough:

  ```
  Sample generation:
    Prompt: Share tips to learn python
    Persona: tutor
    Response: From a balanced tutor perspective, let's look at "Share tips to learn python." This ties closely to the "how_to" intent I detected. Consider journaling about: Study Schedule (Plan), Focus Blocks (Practice), Break Strategies (Reflect). Try to plan Study Schedule, practice Focus Blocks, and reflect on Break Strategies. Knowledge focus: schedule focused practice blocks. Related concepts worth exploring: work through bite-sized python projects, review core syntax and standard library patterns, reflect on debugging takeaways.
    Journaling topics: Study Schedule, Focus Blocks, Break Strategies
    Knowledge concepts: schedule focused practice blocks, work through bite-sized python projects, review core syntax and standard library patterns, reflect on debugging takeaways
    Knowledge scores: relevance=3.956, coverage=0.865, cohesion=0.776, collaboration=0.349, diversity=6.867, K_raw=0.829, gate_mean=0.736

  Calibration report: ECE raw=0.437 -> calibrated=0.027 (reduction=94%)
  Reward weights: [0.2666619 0.2923091 0.075     0.366029 ]

  Intent bandit walkthrough:
  Prompt: Clarify when to use breadth-first search
  Classifier intent: definition (optimal=definition)
  Reward components: correctness=1.00, confidence=1.00, semantic=0.80, feedback=0.92
  Composite reward: 0.96
  Response: use case → shortest path in unweighted graphs; contrasts with → depth-first search

  Prompt: How should I start researching renewable energy?
  Classifier intent: how_to (optimal=how_to)
  Reward components: correctness=1.00, confidence=0.45, semantic=0.80, feedback=0.92
  Composite reward: 0.80
  Response: first step → audit local energy use; research → read government energy outlook

  Prompt: Compare supervised and unsupervised learning
  Classifier intent: comparison (optimal=comparison)
  Reward components: correctness=1.00, confidence=1.00, semantic=0.84, feedback=0.92
  Composite reward: 0.96
  Response: compare with → unsupervised learning; focus → labeled data; focus → pattern discovery

  Prompt: Offer reflective prompts for creative writing
  Classifier intent: exploration (optimal=exploration)
  Reward components: correctness=1.00, confidence=0.42, semantic=0.80, feedback=0.92
  Composite reward: 0.79
  Response: prompt → explore character motivations; prompt → reflect on sensory details
  ```

  The quickstart rewards are simulated using the intent classifier's posterior probabilities so the bandit loop stays in the unit
  interval without external feedback.

  You can opt into saving the calibrated accuracy curve and the empirical-vs-theoretical EXP3 regret comparison that back the
  analysis appendix by setting `SEMANTIC_LEXICON_SAVE_PLOTS=1` (or `true/yes/on`) before running the script. This keeps the
  repository free of bulky PNGs by default while still letting you regenerate them under `docs/assets/` on demand. Refer to the
  generated CSV summaries in `Archive/` for the underlying values if you wish to recreate the plots with your preferred tooling.
  The same behaviour is available through the CLI:

### Fixed-Point Ladders Companion

The research brief that motivated the README examples now has a full mathematical companion in
[`docs/articles/fixed-point-ladders.md`](docs/articles/fixed-point-ladders.md). The article walks through:

- **Parts A–C (Foundations & Logic):** proofs of the lattice background, the Knaster–Tarski theorem, Kleene iteration, and
  µ-calculus semantics, all illustrated with the reachability operator that powers the persona-aware knowledge search.
- **Parts D–H (Shortcuts & Optimisation):** contraction-based accelerations, closure operators for finite-time stabilisation,
  and multi-objective "best layer" selection rules that mirror the reward-shaping heuristics used in the quickstart bandit demo.
- **Parts I (Reflection):** a diagrammatic summary that ties the layer-by-layer iterations back to the automation loops in this
  repository, making it easy to map the abstract ladders onto concrete CLI behaviours.

Each section keeps the ladder visual from the README and annotates it with the corresponding proofs or calculations so the
math-heavy readers can cross-check the guarantees while experimenting with the code.

  For a dedicated, math-forward treatment of the fixed-point ladders referenced above, consult
  [docs/articles/fixed-point-ladders.md](docs/articles/fixed-point-ladders.md).

```bash
semantic-lexicon generate "What is machine learning?" \
  --workspace artifacts \
  --persona tutor \
  --config config.yaml
```

## Cross-domain validation & profiling

Run the bundled validation harness to stress-test the calibrated intent router on
100 prompts that span science, humanities, business, wellness, and personal
development queries:

```bash
PYTHONPATH=src python examples/cross_domain_validation.py
```

The script trains the classifier, evaluates it on the new prompt set, and saves a
report to `Archive/cross_domain_validation_report.json`. We report
\(\mathrm{Last}(\mathcal{R})\) and the corresponding content address \(h_{j^\star}\)
as defined in §4.

Runs are archived in `Archive/topic_pure_retrieval_runs.json`.

### §1 Core objects

```math
\begin{aligned}
&\mathbb{R}^d,\ d\in\mathbb{N}.\\
&\mathcal{C}=\{c_i\}_{i=1}^{N},\ E[c]\in\mathbb{R}^d.\\
&z:\mathcal{Q}\to \mathbb{R}^d.\\
&p\in\mathbb{R}^d\ \text{(use } p=0 \text{ if not applicable)}.\\
&g\in[0,1]^d,\quad M\succeq 0\in\mathbb{R}^{d\times d}.\\
&W=\Sigma^{-1/2}.
\end{aligned}
```

#### Scoring and retrieval

```math
r(q)=\mathrm{diag}(g)\,\big(z(q)+p\big),\qquad
s(q,c)=\big(Wr(q)\big)^{\!\top} M \big(WE[c]\big),\qquad
S_k(q)=\mathop{\mathrm{arg\,topk}}_{c\in\mathcal{C}} s(q,c).
```

### §2 Evaluation archive and identifiers

```math
R_j=\big(\Theta_j,\ \mathcal{D}_j,\ \mathcal{T}_j,\ \mathbf{m}_j,\ t_j\big).
```

```math
h_j=\mathsf{H}\!\big(\Theta_j,\,\mathcal{D}_j,\,\mathcal{T}_j\big),\quad
\text{with } \mathsf{H}:{\{0,1\}^\ast}\to\{0,1\}^{256} \text{ collision-resistant.}
```

### §3 Metrics

```math
\text{Purity@}k(q)=\frac{1}{k}\sum_{c\in S_k(q)}\mathbf{1}\{y(c)=y(q)\}.
```

```math
\mathsf{TVR}=\mathbb{P}\big[s(q,c^+)\le s(q,c^-)\big],\qquad
\mathsf{GS}=\frac{\|g\|_0}{d},\qquad
\kappa(\Sigma)=\frac{\lambda_{\max}(\Sigma)}{\lambda_{\min}(\Sigma)}.
```

```math
\mathbf{m}_j=\big(\ \overline{\text{Purity@}5},\ \overline{\text{Purity@}10},\ \mathsf{TVR},\ \mathsf{GS},\ \kappa(\Sigma)\ \big)_j,\quad
\overline{\cdot}\ \text{averages over } \mathcal{D}_j.
```

### §4 README functionals

```math
\mathrm{Last}(\mathcal{R})=\mathbf{m}_{j^\star},\quad j^\star=\arg\max_j t_j.
```

```math
\mathrm{Best}_{f,w}(\mathcal{R})=\mathbf{m}_{\arg\max_j f(w\odot \mathbf{m}_j)},\quad w\in\mathbb{R}_{\ge 0}^m,\ f:\mathbb{R}^m\to\mathbb{R}\ \text{monotone}.
```

```math
\mathrm{Mean}(\mathcal{R})=\frac{1}{n}\sum_{j=1}^{n}\mathbf{m}_j,\quad n=|\mathcal{R}|.
```

```math
\Delta(\mathcal{R})=\mathbf{m}_{j^\star}-\mathbf{m}_{j^\star-1}\quad (\text{defined if } n\ge 2).
```

```math
\big(h_{j^\star},\,\mathrm{Last}(\mathcal{R})\big)=\Big(\mathtt{845d7c3479535bdc83f7ed403e5b3695f242cc4561c807421f5c70d0c941291b},\ (0.6,0.5,0.0,1.0,371.6768300721485)\Big).
```

### §5 Example prompt I/O

```math
\mathcal{Q}^\star=\{q_0,q_1,q_2,q_3\},\qquad k=2.
```

```math
\Pi_k(q)=\big(S_k(q),\ s(q, S_k(q))\big).
```

```math
\text{Examples}(\mathcal{Q}^\star;\ h)=\Big\{\,\big(q,\ \Pi_k^{(h)}(q)\big)\ :\ q\in \mathcal{Q}^\star \Big\},\quad h=\mathtt{845d7c3479535bdc83f7ed403e5b3695f242cc4561c807421f5c70d0c941291b}.
```

### §6 Guarantees

- Lossless history: \(j\mapsto R_j\) is injective; the README exposes \(\{\mathbf{m}_j\}\) via \((h_{j^\star},\mathrm{Last}(\mathcal{R}))\).
- Determinism: for fixed \(h\) and \(q\), \(\Pi_k^{(h)}(q)\) is unique.
- Stability: \(P(P(M))=P(M),\ M\succeq 0\Rightarrow P(M)=M,\ P(M)\succeq 0\).

A companion benchmark is written to `Archive/intent_performance_profile.json`.
With heuristic fast paths, sparse dot products, and vector caching enabled the
optimised classifier processes repeated prompts **60 % faster** than the baseline
float64 pipeline (1.83 ms → 0.73 ms per request) while keeping the same accuracy.
Caching retains the most recent vectors, so the optimised pipeline uses ~27 KB of
RAM versus the baseline’s 4 KB; the additional footprint is documented alongside
the latency numbers so deployments can choose the appropriate trade-off.

## Streaming feedback API

Real-time user feedback can be folded into the composite reward with the new
HTTP server. Launch the background service by wiring an `IntentClassifier`
through `FeedbackService` and `FeedbackAPI`:

```python
from semantic_lexicon import IntentClassifier, IntentExample
from semantic_lexicon.api import FeedbackAPI, FeedbackService
from semantic_lexicon.utils import read_jsonl

examples = [
    IntentExample(text=str(rec["text"]), intent=str(rec["intent"]), feedback=0.92)
    for rec in read_jsonl("src/semantic_lexicon/data/intent.jsonl")
]
classifier = IntentClassifier()
classifier.fit(examples)
service = FeedbackService(classifier)
api = FeedbackAPI(service, host="127.0.0.1", port=8765)
api.start()
```

Submit streaming feedback with a simple POST request:

```bash
curl -X POST http://127.0.0.1:8765/feedback \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Compare supervised and unsupervised learning", \
        "selected_intent": "comparison", \
        "optimal_intent": "comparison", \
        "feedback": 0.96}'
```

The server replies with the updated composite-reward weights and the component
vector that was logged. Each event is processed under a lock so parallel clients
can stream feedback without clobbering the learned weights, and the new reward
weights remain simplex-projected for EXP3 compatibility.

Key parameters for `semantic-lexicon generate`:

- `--workspace PATH` – directory that contains the trained embeddings and weights (defaults to `artifacts`).
- `--persona NAME` – persona to blend into the response (defaults to the configuration's `default_persona`).
- `--config PATH` – optional configuration file to override model hyperparameters during loading.

## Adversarial Style Selection

Semantic Lexicon now bundles EXP3 helpers for experimenting with
adversarial persona *and* intent selection. The following snippet alternates
between two personas while learning from scalar feedback in ``[0, 1]``:

```python
from semantic_lexicon import AnytimeEXP3, NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig

config = SemanticModelConfig()
model = NeuralSemanticModel(config)
trainer = Trainer(model, TrainerConfig())
trainer.train()

bandit = AnytimeEXP3(num_arms=2)
personas = ["tutor", "researcher"]

for prompt in [
    "Outline matrix factorisation for recommendations",
    "Give journaling prompts about creativity",
    "Explain reinforcement learning trade-offs",
]:
    arm = bandit.select_arm()
    persona = personas[arm]
    response = model.generate(prompt, persona=persona)
    score = min(1.0, len(response.response.split()) / 40.0)
    bandit.update(score)
```

### Intent Selection with EXP3

We can model intent routing as an adversarial bandit problem. Let ``K`` be
the number of intents (e.g. ``{"how_to", "definition", "comparison", "exploration"}``).
At round ``t`` the system receives a prompt ``P_t`` and chooses an intent ``I_t``
using EXP3. After delivering the answer, a reward ``r_t`` in ``[0, 1]`` arrives
from explicit ratings or engagement metrics. The arm-selection probabilities are

$$
p_i(t) = (1 - \gamma) \frac{w_i(t)}{\sum_{j=1}^{K} w_j(t)} + \frac{\gamma}{K},
$$

and the weight for the played intent updates via

$$
w_{I_t}(t+1) = w_{I_t}(t) \exp\left(\frac{\gamma r_t}{K p_{I_t}(t)}\right).
$$

When the horizon ``T`` is unknown, the bundled ``AnytimeEXP3`` class applies the
doubling trick to refresh its parameters so the regret remains ``O(\sqrt{T})``.

The quickstart script demonstrates the pattern by mapping arms to intent labels
and simulating rewards from the classifier's posterior probability:

```python
from semantic_lexicon import AnytimeEXP3, NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig

config = SemanticModelConfig()
model = NeuralSemanticModel(config)
trainer = Trainer(model, TrainerConfig())
trainer.train()

intents = [label for _, label in sorted(model.intent_classifier.index_to_label.items())]
bandit = AnytimeEXP3(num_arms=len(intents))
prompt = "How should I start researching renewable energy?"
arm = bandit.select_arm()
intent = intents[arm]
   reward = model.intent_classifier.predict_proba(prompt)[intent]
   bandit.update(reward)
   ```

## Intent-Bandit Analysis Toolkit

The `semantic_lexicon.analysis` module supplies the maths underpinning the
improved EXP3 workflow:

- `RewardComponents` & `composite_reward` combine correctness, calibration,
  semantic, and feedback signals into the bounded reward required by EXP3.
- `estimate_optimal_weights` fits component weights via simplex-constrained least
  squares on historical interactions.
- `DirichletCalibrator` provides Bayesian confidence calibration with a
  Dirichlet prior, yielding posterior predictive probabilities that minimise
  expected calibration error.
- `simulate_intent_bandit` and `exp3_expected_regret` numerically check the
  \(2.63\sqrt{K T \log K}\) regret guarantee for the composite reward.
- `compute_confusion_correction` and `confusion_correction_residual` extract the
  SVD-based pseudoinverse that reduces systematic routing errors.
- `RobbinsMonroProcess` and `convergence_rate_bound` expose the stochastic
  approximation perspective with an \(O(1/\sqrt{n})\) convergence rate bound.

See [docs/analysis.md](docs/analysis.md) for full derivations and proofs.

### Intent Classification Objective

Ethical deployment requires robust intent understanding. Semantic Lexicon's
``IntentClassifier`` treats intent prediction as a multinomial logistic regression
problem over prompts ``(P_i, I_i)``. Given parameters ``\theta``, the model
minimises the cross-entropy loss

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log p(I_i \mid P_i; \theta),
$$

which matches the negative log-likelihood optimised during training. Improving
intent accuracy directly translates into higher-quality feedback for the bandit
loop.

## Configuration

Semantic Lexicon reads configuration files in YAML or JSON using the `SemanticModelConfig` dataclass. Example `config.yaml`:

```yaml
embeddings:
  dimension: 50
  max_words: 5000
intent:
  learning_rate: 0.2
  epochs: 5
knowledge:
  max_relations: 4
persona:
  default_persona: tutor
generator:
  temperature: 0.7
```

Load the configuration via CLI (`semantic-lexicon train --config config.yaml`) or programmatically:

```python
from semantic_lexicon import NeuralSemanticModel, load_config

config = load_config("config.yaml")
model = NeuralSemanticModel(config)
```

## Training API

```python
from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig

config = SemanticModelConfig()
model = NeuralSemanticModel(config)
trainer = Trainer(model, TrainerConfig())
trainer.train()
response = model.generate("How to learn python?", persona="tutor")
print(response.response)
```

## Diagnostics Programmatically

```python
from semantic_lexicon.model import NeuralSemanticModel
from semantic_lexicon.training import Trainer, TrainerConfig

model = NeuralSemanticModel()
trainer = Trainer(model, TrainerConfig())
trainer.train()
report = trainer.run_diagnostics()
print(report.to_dict())
```

## Development Workflow

| Task            | Command                           |
| --------------- | --------------------------------- |
| Format & lint   | `ruff check .` · `black .`        |
| Type check      | `mypy src`                        |
| Run tests       | `pytest`                          |
| Preview docs    | `mkdocs serve`                    |

A `Makefile` (or CI workflow) can orchestrate the tasks:

```bash
make lint
make test
make docs
```

## Streams & Clipboard

Generation now distinguishes abstract sources via the prompt functor \(𝐅\). Use cases:
- **Literal prompts** – pass a string and the CLI behaves exactly as before.
- **Streaming prompts** – pass `"-"` to fold STDIN chunks until EOF, perfect for shell pipelines.
- **Clipboard prompts** – call `semantic-lexicon clipboard` to pull the current system clipboard.

Example invocations:

```bash
echo "What is a transformer?" | semantic-lexicon generate - --workspace artifacts
semantic-lexicon clipboard --workspace artifacts --persona exploration
```

Both paths reuse the existing workspace/persona/config pipeline and reject empty inputs with a friendly error.

Sample outputs:

```text
$ echo "What is a transformer?" | semantic-lexicon generate - --workspace artifacts
Persona: generic
Response:
1. Transformers rely on self-attention so tokens draw context from the entire sentence in one step.
2. Multi-head attention lets the model track different relationships (syntax, long-range cues) simultaneously.
3. Decoder layers reuse the same mechanism to generate fluent text token by token.

$ semantic-lexicon clipboard --workspace artifacts --persona exploration
Clipboard prompt: "Give me three research angles on causal discovery."
Persona: exploration
Response:
1. Explore score-based causal discovery that leverages diffusion models to recover graph structure from noise.
2. Compare invariant risk minimisation versus meta-learning for handling interventions and domain shift.
3. Prototype active experimentation loops that query the system for the most informative interventions next.
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install development dependencies: `pip install .[dev]`.
3. Run `make test` to ensure linting, typing, and tests pass.
4. Submit a pull request with detailed notes on new features or fixes.

## Acknowledgments

This work was shaped by the survey "Interpretation of Time-Series Deep Models: A Survey" [(arXiv:2305.14582)](https://arxiv.org/abs/2305.14582) shared by Dr. Zhao after reading our preprint on Calibrated "Counterfactual Conformal Fairness" (C3F) [(arXiv:2509.25295)](https://arxiv.org/abs/2509.25295). His survey offered both the conceptual framing and motivation for exploring this research path. We also thank Hamdi Alakkad and Bugra Kilictas for their pivotal contributions to our related preprints, which laid the groundwork for the developments presented here. We further acknowledge DeepSeek, whose advanced mathematical reasoning and logical inference capabilities substantially enhanced the precision and efficiency of the formal logic analysis, and the collaboration between OpenAI and GitHub on Codex, whose code generation strengths, in concert with DeepSeek’s systems, significantly accelerated and sharpened the overall development and analysis process.

## Author's Note
Hello people, or a system running perfectly, inbetween or broken -- At least working. -- While I am building groups, it is nice to see you behind them. This project represents my core self. We all came from a fixed point and would end up there as well. I am working on making myself “us,” me “our.” The physical world is for receiving and giving feelings, while the symbolic world is the projection of those feelings. Today is October 13, 2025, and I am located in Meckenheim, Germany. My plane landed yesterday from Istanbul—a nice trip, though (p.s. @farukalpayy). So, did you all feel like the energy was broken? It was the point where you get deep enough to realize where it was going. We reached the point where f(x) = x holds, but f(x) = y itself is also a point. And at this point, my request could be clarified. If this project saves you time or money, please consider sponsoring. Most importantly, it helps me keep improving and offering it free for the community. [Visit my Donation Page](https://buymeacoffee.com/farukalpay)

## Contact & Legal

- Semantic Lexicon is a Lightcap® research project distributed as open source under the Apache License 2.0; see [LICENSE](LICENSE) for details on rights and obligations.
- Lightcap® is a registered trademark (EUIPO Reg. No. 019172085).
- For enquiries, contact [alpay@lightcap.ai](mailto:alpay@lightcap.ai).
