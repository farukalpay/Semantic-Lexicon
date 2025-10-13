# Prompt Evaluation Hub

This section curates both the manual and automated evaluations of the tutor persona so
contributors can trace how prompts are exercised across disciplines, tones, and
learning goals.

## 100-Prompt Automated Sweep

- **Coverage:** 10 thematic clusters (mathematics, physics, chemistry, biology,
  history, literature, philosophy, business, technology, wellness) with ten prompts each.
- **Persona:** `tutor`
- **Output format:** JSON containing the prompt, generated reply, detected intents,
  surfaced topics, knowledge graph hits, and an agent-assigned score with rationale.
- **Scoring transparency:** Each entry logs `score = 2.5 base + 0.02*word_count +
  0.3*topics(up to 3) + 0.2*knowledge_hits(up to 2)` alongside the raw counts so the
  capped contributions are explicit.

Download the full run from `tests/prompt_generation_report.json`.

## Manual Spot Check

For qualitative commentary on the first six tutor generations—covering set-up,
observations, and recommended follow-ups—see
[`manual_prompt_test.md`](./manual_prompt_test.md).

## Reproducing the Sweep

1. Install the project in editable mode (`pip install -e .`).
2. Export `PYTHONPATH=src` so the bundled modules can be imported.
3. Run the evaluation script:

   ```bash
   PYTHONPATH=src python docs/scripts/run_prompt_sweep.py
   ```

   The helper script mirrors the configuration used for the published JSON (training the
   miniature model and iterating through the 100 curated prompts).

