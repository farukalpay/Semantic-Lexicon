from pathlib import Path

from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.data import load_sample_corpus
from semantic_lexicon.training import CorpusProcessor, Trainer


def test_model_generates_response(tmp_path: Path) -> None:
    config = SemanticModelConfig(workspace=tmp_path)
    model = NeuralSemanticModel(config=config)
    response = model.generate_response("Explain neural networks", persona="tutor")
    assert response


def test_training_pipeline(tmp_path: Path) -> None:
    config = SemanticModelConfig(workspace=tmp_path)
    processor = CorpusProcessor()
    processed = processor.prepare_corpus([{"title": "Neural", "summary": "Neural models"}])
    trainer = Trainer(NeuralSemanticModel(config=config), config=config.trainer)
    trainer.train(processed)
    artifacts = trainer.model.save(tmp_path)
    assert artifacts.embeddings_path.exists()
