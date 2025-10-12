"""High-level workflows that combine corpus preparation and training."""

from __future__ import annotations

from typing import Iterable, MutableMapping

from ..config import SemanticModelConfig
from ..model import NeuralSemanticModel
from ..training import CorpusProcessor, Trainer


def prepare_and_train(entries: Iterable[MutableMapping[str, object]], config: SemanticModelConfig | None = None) -> NeuralSemanticModel:
    """Prepare ``entries`` and train a :class:`NeuralSemanticModel`."""
    config = config or SemanticModelConfig()
    processor = CorpusProcessor()
    processed = processor.prepare_corpus(entries)
    model = NeuralSemanticModel(config=config)
    trainer = Trainer(model, config=config.trainer)
    trainer.train(processed)
    model.save(config.workspace)
    return model
