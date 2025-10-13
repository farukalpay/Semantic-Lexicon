# CLI Fresh Start Test (2025-10-13)

This report documents a full command-line test of the Semantic Lexicon toolkit executed in a clean virtual environment.

## Environment Setup
- Created and activated a Python 3 virtual environment via `python -m venv fresh_test_env`.
- Installed the project in editable mode with development and documentation extras using `pip install -e .[dev,docs]`.

## Corpus Preparation
- Generated a fresh workspace directory (`fresh_artifacts`) and populated it with processed intent and knowledge corpora by running:
  ```bash
  semantic-lexicon prepare --intent-path src/semantic_lexicon/data/intent.jsonl \
      --knowledge-path src/semantic_lexicon/data/knowledge.jsonl \
      --workspace fresh_artifacts
  ```
  The CLI confirmed that the corpus was prepared successfully.

## Model Training
- Trained the intent classifier and knowledge network with `semantic-lexicon train --workspace fresh_artifacts`.
- Training logs report four intents and eighteen knowledge entities with artifacts saved to the workspace.

## Diagnostics Summary
- Executed `semantic-lexicon diagnostics --workspace fresh_artifacts --output fresh_diagnostics.json`.
- Key metrics from the console output:
  - Intent evaluation matched expected labels for three reference prompts with confidence 1.0 each.
  - Example generations surfaced mixed-relevance concepts (e.g., photosynthesis for a breadth-first search clarification) indicating that knowledge retrieval remains noisy in this configuration.

## Response Generation Spot Checks
- Ran three prompts through `semantic-lexicon generate` with the `tutor` persona:
  1. "Clarify when to use breadth-first search" → classified as `definition`, but suggested off-topic topics such as photosynthesis.
  2. "How should I start researching renewable energy?" → classified as `how_to`, with partially relevant suggestions like auditing local energy use.
  3. "Compare supervised and unsupervised learning" → classified as `definition` instead of `comparison`, though recommended machine learning-related focus areas.

## Quickstart Script
- Executed `PYTHONPATH=src python examples/quickstart.py` to exercise the full quickstart flow.
- The walkthrough reported composite rewards between 0.79 and 0.96 with correct intents for all prompts except the manual CLI run noted above.
- Calibration improved from 0.437 raw ECE to 0.027 after calibration.

## Cross-Domain Validation
- The 100-prompt validation set referenced in student materials was not available in the repository, so cross-domain validation was not performed.

## Cleanup
- Removed transient artifacts (`fresh_artifacts`, virtual environment directory, egg-info, and `__pycache__` folders) after verifying the results to leave the repository clean.

## Observations
- The training and diagnostics pipeline runs end-to-end without errors in a clean environment.
- Intent predictions are stable in diagnostics and the quickstart walkthrough, but manual generation for the BFS prompt still surfaced unrelated concepts, suggesting further tuning of topic retrieval may be needed despite correct intent classification.

