from __future__ import annotations

from semantic_lexicon.config import GeneratorConfig
from semantic_lexicon.generator import GenerationResult, PersonaGenerator
from semantic_lexicon.persona import PersonaStore


def test_generator_returns_result(config) -> None:
    store = PersonaStore()
    persona = store.get("tutor")
    generator = PersonaGenerator(GeneratorConfig())
    result = generator.generate("Explain AI", persona, ["definition"])
    assert isinstance(result, GenerationResult)
    assert isinstance(result.response, str)
    assert result.response
