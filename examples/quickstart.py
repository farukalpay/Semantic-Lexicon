"""Minimal quickstart script for Semantic Lexicon.

The script trains the bundled miniature model, demonstrates the calibrated
intent routing loop with composite rewards, and saves analysis plots that
compare empirical EXP3 regret against the theoretical bound.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from semantic_lexicon import (
    NeuralSemanticModel,
    SemanticModelConfig,
    composite_reward,
    exp3_expected_regret,
    simulate_intent_bandit,
)
from semantic_lexicon.utils import read_jsonl
from semantic_lexicon.training import Trainer, TrainerConfig

GROUND_TRUTH = {
    "Clarify when to use breadth-first search": "definition",
    "How should I start researching renewable energy?": "how_to",
    "Compare supervised and unsupervised learning": "comparison",
    "Offer reflective prompts for creative writing": "exploration",
}

FEEDBACK = {
    prompt: 0.92 for prompt in GROUND_TRUTH
}

KNOWLEDGE_RECORDS = list(read_jsonl(Path("src/semantic_lexicon/data/knowledge.jsonl")))
KNOWLEDGE_EDGES: dict[str, list[tuple[str, str]]] = {}
for record in KNOWLEDGE_RECORDS:
    KNOWLEDGE_EDGES.setdefault(str(record["head"]), []).append((str(record["relation"]), str(record["tail"])))


def run_generation_demo(model: NeuralSemanticModel) -> None:
    """Generate a single response to confirm the model is trained."""

    response = model.generate("Share tips to learn python", persona="tutor")
    print("Sample generation:\n", response.response, sep="")


def run_intent_bandit(model: NeuralSemanticModel) -> None:
    """Showcase EXP3-driven intent selection for a small prompt batch."""

    classifier = model.intent_classifier
    intents = [label for _, label in sorted(classifier.index_to_label.items())]
    prompts = list(GROUND_TRUTH.keys())

    raw_ece, calibrated_ece = classifier.calibration_report
    print(
        "\nCalibration report: "
        f"ECE raw={raw_ece:.3f} -> calibrated={calibrated_ece:.3f} "
        f"(reduction={(raw_ece - calibrated_ece) / raw_ece:.0%})"
    )
    print("Reward weights:", classifier.reward_weights)

    print("\nIntent bandit walkthrough:")
    for prompt in prompts:
        predicted_intent = classifier.predict(prompt)
        arm = intents.index(predicted_intent)
        optimal_intent = GROUND_TRUTH[prompt]
        feedback = FEEDBACK[prompt]
        components = classifier.reward_components(prompt, predicted_intent, optimal_intent, feedback)
        reward_value = composite_reward(components, classifier.reward_weights)
        response = build_response(prompt, predicted_intent)
        print(
            f"Prompt: {prompt}\n"
            f"Classifier intent: {predicted_intent} (optimal={optimal_intent})\n"
            f"Reward components: correctness={components.correctness:.2f}, "
            f"confidence={components.confidence:.2f}, semantic={components.semantic:.2f}, "
            f"feedback={components.feedback:.2f}\n"
            f"Composite reward: {reward_value:.2f}\n"
            f"Response: {response}\n"
        )


def build_response(prompt: str, intent: str) -> str:
    prompt_lower = prompt.lower()
    suggestions: list[str] = []
    for head, edges in KNOWLEDGE_EDGES.items():
        if head in prompt_lower:
            for relation, tail in edges:
                suggestions.append(f"{relation.replace('_', ' ')} → {tail}")
    if not suggestions:
        if intent == "definition":
            suggestions = ["define key terms", "link to contrasting concepts", "note canonical use cases"]
        elif intent == "how_to":
            suggestions = ["outline actionable steps", "list required resources", "set an initial milestone"]
        elif intent == "comparison":
            suggestions = ["list shared traits", "highlight unique features", "summarise trade-offs"]
        else:
            suggestions = ["brainstorm related themes", "collect open-ended questions", "reflect on motivations"]
    return "; ".join(suggestions[:3])


def save_analysis_plots(model: NeuralSemanticModel) -> None:
    """Persist the convergence and regret plots used in the documentation."""

    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    classifier = model.intent_classifier
    epochs = np.arange(1, len(classifier.training_accuracy_curve) + 1)
    accuracies = classifier.training_accuracy_curve

    if accuracies:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, accuracies, marker="o", color="#2563eb")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Intent accuracy convergence")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(assets / "intent_convergence.png", dpi=200)
        plt.close(fig)

    intents = [label for _, label in sorted(classifier.index_to_label.items())]
    reward_sequences = []
    optimal_indices = []
    horizon = 1000
    for round_idx in range(horizon):
        prompt = list(GROUND_TRUTH.keys())[round_idx % len(GROUND_TRUTH)]
        optimal_intent = GROUND_TRUTH[prompt]
        feedback = FEEDBACK[prompt]
        reward_sequences.append(
            [
                classifier.reward_components(prompt, intent, optimal_intent, feedback)
                for intent in intents
            ]
        )
        optimal_indices.append(intents.index(optimal_intent))

    weights = classifier.reward_weights
    regret_result = simulate_intent_bandit(
        reward_sequences,
        optimal_indices,
        weights,
        rng=np.random.default_rng(11),
    )
    bound = exp3_expected_regret(len(intents), horizon)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, horizon + 1), regret_result.regret, label="Empirical regret", color="#10b981")
    ax.axhline(bound, color="#ef4444", linestyle="--", label="Theoretical bound")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("EXP3 regret vs theoretical bound")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(assets / "exp3_regret.png", dpi=200)
    plt.close(fig)


def main() -> None:
    config = SemanticModelConfig()
    model = NeuralSemanticModel(config)
    trainer = Trainer(model, TrainerConfig())
    trainer.train()

    run_generation_demo(model)
    run_intent_bandit(model)
    save_analysis_plots(model)


if __name__ == "__main__":
    main()
