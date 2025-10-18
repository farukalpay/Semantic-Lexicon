# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

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


def test_generator_handles_structured_matrix_prompt() -> None:
    store = PersonaStore()
    persona = store.get("tutor")
    generator = PersonaGenerator(GeneratorConfig())
    prompt = (
        "You are a math solver. Do the concrete computations only. Task: "
        "Let S=[[2, 0], [0, 1]] (scale x by 2) and R=[[0, -1], [1, 0]] "
        "(rotate 90° CCW). 1) Compute RS and RS*(1,0). 2) Compute SR and SR*(1,0). "
        "Return markdown with exactly these sections: ## Matrices, ## Composition, ## Results."
    )
    result = generator.generate(prompt, persona, ["definition"])
    assert "## Matrices" in result.response
    assert "RS = R × S" in result.response
    assert "RS · v" in result.response
    assert result.knowledge_hits == []
    assert result.phrases == []


def test_generator_handles_section_trigger_case_insensitively() -> None:
    store = PersonaStore()
    persona = store.get("tutor")
    generator = PersonaGenerator(GeneratorConfig())
    prompt = (
        "return markdown with exactly these sections: ## Matrices, ## Composition, ## Results. "
        "Given S=[[1, 1], [0, 1]] and R=[[0, 1], [-1, 0]], and v=(1, 0)."
    )
    result = generator.generate(prompt, persona, ["definition"])
    assert "## Matrices" in result.response
    assert "SR = S × R" in result.response
    assert "SR · v" in result.response
