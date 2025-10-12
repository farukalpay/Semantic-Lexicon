from semantic_lexicon import NeuralSemanticModel
from semantic_lexicon.diagnostics import run_all_diagnostics


def test_diagnostics_runs() -> None:
    model = NeuralSemanticModel()
    result = run_all_diagnostics(model=model)
    assert result.embedding_stats
    assert result.generations
