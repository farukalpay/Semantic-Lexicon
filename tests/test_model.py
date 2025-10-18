from __future__ import annotations

from semantic_lexicon.model import NeuralSemanticModel


def test_generate_handles_literal_prompts_without_training() -> None:
    model = NeuralSemanticModel()

    result = model.generate("Return only the number 7, nothing else.")

    assert result.response.strip() == "7"
    assert isinstance(result.intents, list)
