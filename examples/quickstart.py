"""Minimal quickstart script for Semantic Lexicon."""

from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig


def main() -> None:
    config = SemanticModelConfig()
    model = NeuralSemanticModel(config)
    trainer = Trainer(model, TrainerConfig())
    trainer.train()
    response = model.generate("Share tips to learn python", persona="tutor")
    print(response.response)


if __name__ == "__main__":
    main()
