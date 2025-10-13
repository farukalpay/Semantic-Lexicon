# Manual Prompt Test Report

## Overview
- **Date:** 2025-10-13
- **Model setup:** Trained bundled Semantic Lexicon components via `Trainer` with default `SemanticModelConfig` and `TrainerConfig`.
- **Persona:** `tutor`
- **Runtime logs:** Training completed for intent classifier and knowledge network prior to generation.

## Prompt Runs
| # | Prompt | Detected Intent | Suggested Topics | Notes |
|---|--------|-----------------|------------------|-------|
| 1 | Outline steps to bake sourdough bread. | how_to | Public Speaking (Explore); Practice Routine (Practice); Feedback Loops (Reflect) | Intent detection aligns with procedural request, but topic set is unrelated to baking. |
| 2 | Summarise the theory of relativity. | definition | Photosynthesis (Define); Chlorophyll Function (Explore); Energy Conversion (Connect) | Detected intent matches summary/definition ask; topics are unrelated scientific concepts. |
| 3 | Give me tips for daily mindfulness. | how_to | Study Schedule (Plan); Focus Blocks (Practice); Break Strategies (Reflect) | Output stays productivity-focused; partially relevant but lacks mindfulness-specific guidance. |
| 4 | Define quantum entanglement. | definition | Potential Energy (Define); Reference Frames (Explore); Energy Transfer (Connect) | Intent classification is on target; knowledge suggestions stay within physics framing but not entanglement-specific. |
| 5 | What are key stages in project management? | definition | Machine Learning (Define); Supervised Learning (Explore); Generalization Error (Compare) | Intent matches inquiry, yet recommended topics diverge into ML concepts. |
| 6 | Compare supervised and unsupervised learning. | definition | Photosynthesis (Define); Supervised Learning (Explore); Energy Conversion (Connect) | Mixed topicality: partially correct (Supervised Learning) but includes unrelated biology references. |

## Observations
- **Strengths:** Intent classifier consistently distinguished between procedural (how_to) and explanatory (definition) requests.
- **Limitations:** Knowledge/topic suggestions often recycled mismatched entries from the compact sample dataset, leading to low topical fidelity for domain-specific prompts.
- **Next steps:** Consider expanding the sample knowledge graph and embeddings with domain-appropriate concepts to improve topical coverage. Additional fine-tuning or prompt-conditioned reranking could further align suggested topics.
