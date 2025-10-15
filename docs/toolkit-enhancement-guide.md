# Toolkit Enhancement Playbook

This playbook collects concrete upgrades that improve the quality of
Semantic Lexicon outputs. Each section ties a recommendation to the CLI
or Python APIs so the workflow stays reproducible.

## Scale-up recommendations (next)

- **Data.** Target 4–6 intents with 50–100 labelled examples each.
  Sample domain prompts via `scripts/sample_intents.py --n 20` and spot
  check class balance before annotating.
- **Knowledge.** Curate 100–300 triples per domain. Validate with
  `jq 'select(.relation=="regulates")'` to catch conflicts before
  loading.
- **Capacity.** Set embeddings to 48 dims for baseline runs; jump to 64
  with `semantic-lexicon train --config configs/scale64.yaml` when
  latency budgets allow.
- **Evaluation.** After each retrain run
  `semantic-lexicon diagnostics --workspace <run>` and check
  `outputs/diagnostics.json` for coverage drift.
- **Reproducibility.** Export `PYTHONHASHSEED=42`, record
  `pip freeze > outputs/requirements.lock`, and log CLI invocations in
  `RUNBOOK.md`.

## 1. Curate a domain-focused corpus

1. **Collect representative prompts.** Pull questions, briefs, and
   troubleshooting logs from the target domain. Normalise casing and
   whitespace with `semantic_lexicon.utils.normalise_text` or the
   `prepare` command so features match the NumPy training loops.
2. **Expand coverage.** Aim for 50–100 labelled prompts per
   intent so the multinomial logistic regression in
   [`IntentClassifier`](../src/semantic_lexicon/intent.py) can estimate
   reliable weights.
3. **Store as JSONL.** Write raw data to a `intent.jsonl` file where
   each line is of the form:

   ```json
   {"text": "Summarise the HIPAA privacy rule", "intent": "definition"}
   {"text": "List steps to onboard a clinical trial site", "intent": "how_to"}
   {"text": "Compare mRNA and viral vector vaccine platforms", "intent": "comparison"}
   {"text": "Brainstorm reflective prompts for patient journaling", "intent": "exploration"}
   ```

   Use domain-specific verbs in the `text` field to make intent signals
   unambiguous.

## 2. Extend the intent inventory

1. **Add new labels.** Introduce intents such as `risk_assessment`,
   `regulation_lookup`, or `tool_recommendation` when the corpus reveals
   repeated behaviours that do not fit the default set.
2. **Document decision rules.** Maintain a short rubric describing when
   each label applies. Store it alongside the dataset to keep
   annotations consistent.
3. **Update decoding templates.** For every intent, create a response
   scaffold in `docs/` or via the prompt templates module so generation
   stays structured (e.g., bullet lists for `how_to`, risk matrices for
   `risk_assessment`).

## 3. Expand the knowledge graph

1. **Encode factual triples.** Append 100–300 domain facts to
   `knowledge.jsonl` using the format:

   ```json
   {"head": "hipaa", "relation": "regulates", "tail": "patient data privacy"}
   {"head": "clinical trial", "relation": "requires", "tail": "informed consent"}
   {"head": "mRNA vaccine", "relation": "advantage", "tail": "rapid redesign"}
   ```

2. **Cluster by theme.** Group triples by topic (compliance, safety,
   methodology) so the `prepare` pipeline can surface cohesive concepts.
3. **Review for conflicts.** Remove or edit generic triples that clash
   with the new domain to avoid polluting the selection scores.

## 4. Calibrate personas

1. **Create persona profiles.** Extend
   [`persona_registry.json`](../src/semantic_lexicon/data/persona_registry.json)
   (or the in-memory defaults) with vectors capturing tone, expertise,
   and risk posture. For example: `"clinical_reviewer"`,
   `"patient_advocate"`, `"compliance_officer"`.
2. **Blend variable attributes.** Use the persona blending utilities to
   interpolate between safety-focused and exploration-focused voices when
   prompts demand nuance.
3. **Audit outputs.** Run `semantic-lexicon generate` for each persona
   on a validation prompt list and verify tone, vocabulary, and caution
   align with the specification.

## 5. Tune training and decoding hyperparameters

1. **Embedding size.** Set `embedding_dim` in `config.yaml` to 48 by
   default (upgrade to 64 if latency allows) when the vocabulary grows
   beyond the sample corpus.
2. **Optimizer steps.** Raise `max_epochs` or adjust `learning_rate`
   within `TrainerConfig` to ensure convergence on the expanded datasets.
3. **Decoding controls.** Experiment with temperature and top-k settings
   in the generator to keep persona-conditioned outputs faithful yet
   diverse. Document the chosen defaults in configuration files.

## 6. Evaluation metrics and logging

1. **Intent accuracy.** Split data into train/validation and report
   accuracy, macro F1, and confusion matrices.
2. **Knowledge retrieval.** Track coverage and cohesion from the
   diagnostics report to validate concept selection quality.
3. **Generation quality.** Score responses with BLEU, ROUGE-L, or
   domain rubrics (e.g., compliance checklists). Store metrics in
   versioned JSON artifacts for reproducibility.

## 7. Reproducibility practices

1. **Control randomness.** Export `PYTHONHASHSEED=42` and set the
   toolkit's seeding utilities before each run.
2. **Freeze dependencies.** Capture `pip freeze > requirements.lock` to
   version the environment alongside datasets.
3. **Record command history.** Keep a `RUNBOOK.md` that lists every CLI
   invocation (prepare/train/diagnostics) with timestamped parameters.

## 8. Hardware and scaling options

1. **Vectorised NumPy.** The training loops already use vectorised
   NumPy; enable BLAS acceleration by installing `numpy` wheels linked to
   OpenBLAS or MKL on the host.
2. **Batch evaluation.** When prompts scale, run `semantic-lexicon
   generate` via the Python API in batches so persona blending happens on
   arrays rather than per-string loops.
3. **Future GPU support.** If porting to CuPy/JAX, replicate the
   deterministic seeds and regularisation schedule documented in
   `analysis.md` to preserve behaviour.

## 9. End-to-end workflow recap

1. Update `intent.jsonl` and `knowledge.jsonl` with the new domain
   entries.
2. Run:

   ```bash
   semantic-lexicon prepare \
     --intent data/domain/intent.jsonl \
     --knowledge data/domain/knowledge.jsonl \
     --workspace artifacts/domain

   semantic-lexicon train --workspace artifacts/domain
   semantic-lexicon diagnostics --workspace artifacts/domain --output diagnostics.json
   ```

3. Evaluate persona-conditioned generations and iterate on personas,
   templates, or hyperparameters based on logged metrics.

Following these steps ensures the enhanced corpus, expanded intents,
refined personas, and tuned hyperparameters translate into measurable
quality gains for the Semantic Lexicon toolkit.
