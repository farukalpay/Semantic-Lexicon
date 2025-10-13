"""Minimal quickstart script for Semantic Lexicon.

The script trains the bundled miniature model and then demonstrates how to
combine the intent classifier with the EXP3 adversarial bandit for intent
selection.  Rewards are simulated using the classifier's posterior
probabilities so the example can run without an external feedback loop.
"""

import numpy as np

from semantic_lexicon import AnytimeEXP3, NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig


def run_generation_demo(model: NeuralSemanticModel) -> None:
    """Generate a single response to confirm the model is trained."""

    response = model.generate("Share tips to learn python", persona="tutor")
    print("Sample generation:\n", response.response, sep="")


def run_intent_bandit(model: NeuralSemanticModel) -> None:
    """Showcase EXP3-driven intent selection for a small prompt batch."""

    intents = [label for _, label in sorted(model.intent_classifier.index_to_label.items())]
    bandit = AnytimeEXP3(num_arms=len(intents), rng=np.random.default_rng(7))
    prompts = [
        "Clarify when to use breadth-first search",
        "How should I start researching renewable energy?",
        "Compare supervised and unsupervised learning",
        "Offer reflective prompts for creative writing",
    ]

    print("\nIntent bandit walkthrough:")
    for prompt in prompts:
        arm = bandit.select_arm()
        chosen_intent = intents[arm]
        reward = model.intent_classifier.predict_proba(prompt)[chosen_intent]
        result = model.generate(prompt, persona="tutor")
        print(
            f"Prompt: {prompt}\n"
            f"Selected intent: {chosen_intent} (reward={reward:.2f})\n"
            f"Response: {result.response}\n"
        )
        bandit.update(reward)


def main() -> None:
    config = SemanticModelConfig()
    model = NeuralSemanticModel(config)
    trainer = Trainer(model, TrainerConfig())
    trainer.train()

    run_generation_demo(model)
    run_intent_bandit(model)


if __name__ == "__main__":
    main()
