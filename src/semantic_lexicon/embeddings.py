"""Embedding subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from .config import EmbeddingConfig
from .logging import configure_logging

LOGGER = configure_logging(logger_name=__name__)


class GloVeEmbeddings:
    """Light-weight wrapper for GloVe-style embeddings."""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig()
        self.word_to_vector: dict[str, np.ndarray] = {}
        self.word_to_index: dict[str, int] = {}
        self.index_to_word: dict[int, str] = {}
        self.embedding_matrix: Optional[np.ndarray] = None

    # Persistence -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "dimension": self.config.dimension,
                "max_words": self.config.max_words,
            },
            "word_to_index": self.word_to_index,
            "embeddings": self.embedding_matrix.tolist() if self.embedding_matrix is not None else [],
        }
        with path.open("w", encoding="utf8") as handle:
            json.dump(payload, handle)
        LOGGER.info("Saved embeddings to %s", path)

    @classmethod
    def load(cls, path: Path) -> "GloVeEmbeddings":
        path = Path(path)
        with path.open("r", encoding="utf8") as handle:
            payload = json.load(handle)
        config = EmbeddingConfig(
            dimension=payload["config"]["dimension"],
            max_words=payload["config"].get("max_words"),
        )
        instance = cls(config=config)
        instance.word_to_index = {str(k): int(v) for k, v in payload["word_to_index"].items()}
        instance.index_to_word = {int(v): str(k) for k, v in instance.word_to_index.items()}
        instance.embedding_matrix = np.asarray(payload["embeddings"], dtype=float)
        for word, index in instance.word_to_index.items():
            instance.word_to_vector[word] = instance.embedding_matrix[index]
        LOGGER.info("Loaded embeddings from %s", path)
        return instance

    # Loading ---------------------------------------------------------------------
    def load_glove(self, filepath: Path) -> None:
        filepath = Path(filepath)
        LOGGER.info("Loading embeddings from %s", filepath)
        vectors: List[np.ndarray] = []
        words: List[str] = []
        dimension = self.config.dimension
        max_words = self.config.max_words
        with filepath.open("r", encoding="utf8") as handle:
            for idx, line in enumerate(handle):
                if max_words is not None and idx >= max_words:
                    break
                pieces = line.strip().split()
                if len(pieces) <= 2:
                    continue
                word, values = pieces[0], pieces[1:]
                vector = np.asarray(values, dtype=float)
                if vector.shape[0] != dimension:
                    continue
                words.append(word)
                vectors.append(vector)
        LOGGER.info("Loaded %d vectors", len(words))
        self._build_matrix(words, vectors)

    def _build_matrix(self, words: List[str], vectors: List[np.ndarray]) -> None:
        vocab_size = len(words) + 1
        self.embedding_matrix = np.zeros((vocab_size, self.config.dimension), dtype=float)
        for index, vector in enumerate(vectors, start=1):
            self.embedding_matrix[index] = vector
        self.word_to_index = {word: i for i, word in enumerate(words, start=1)}
        self.word_to_index["<pad>"] = 0
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.word_to_vector = {word: self.embedding_matrix[index] for word, index in self.word_to_index.items()}

    # Lookup ----------------------------------------------------------------------
    def __contains__(self, word: str) -> bool:
        return word.lower() in self.word_to_index

    def get_embedding(self, word: str) -> np.ndarray:
        word = word.lower()
        if word in self.word_to_index and self.embedding_matrix is not None:
            return self.embedding_matrix[self.word_to_index[word]]
        return np.random.default_rng(abs(hash(word)) % 2**32).normal(0, 0.1, size=self.config.dimension)

    def encode_tokens(self, tokens: Iterable[str]) -> np.ndarray:
        vectors = [self.get_embedding(token) for token in tokens]
        if not vectors:
            return np.zeros((0, self.config.dimension))
        return np.vstack(vectors)
