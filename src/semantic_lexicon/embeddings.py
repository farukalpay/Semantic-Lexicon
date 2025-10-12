"""Embedding utilities for the Semantic Lexicon."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .logging import get_logger
from .utils.random import deterministic_hash

LOGGER = get_logger(__name__)


@dataclass
class EmbeddingState:
    """Persistable embedding state."""

    vocab: Dict[str, int]
    vectors: np.ndarray


class GloVeEmbeddings:
    """Lightweight GloVe loader with caching and persistence."""

    def __init__(self, embedding_dim: int = 50) -> None:
        self.embedding_dim = embedding_dim
        self.vocab: Dict[str, int] = {"<PAD>": 0}
        self._vectors = np.zeros((1, embedding_dim), dtype="float32")

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def load(self, path: Path, *, max_words: Optional[int] = None) -> None:
        """Load embeddings from a text-based GloVe file."""
        LOGGER.info("Loading embeddings from %s", path)
        words: List[str] = []
        vectors: List[np.ndarray] = []
        with path.open("r", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if max_words is not None and index >= max_words:
                    break
                parts = line.strip().split()
                if not parts:
                    continue
                word, *values = parts
                vector = np.asarray(values, dtype="float32")
                if vector.shape[0] != self.embedding_dim:
                    continue
                words.append(word)
                vectors.append(vector)
        self.vocab = {"<PAD>": 0, **{word: i + 1 for i, word in enumerate(words)}}
        padded = np.zeros((len(self.vocab), self.embedding_dim), dtype="float32")
        for word, vector in zip(words, vectors):
            padded[self.vocab[word]] = vector
        self._vectors = padded
        LOGGER.info("Loaded %d embeddings", len(words))

    def get_vector(self, token: str) -> np.ndarray:
        """Return the embedding vector for ``token`` (OOV-safe)."""
        idx = self.vocab.get(token)
        if idx is not None:
            return self._vectors[idx]
        return self._oov_vector(token)

    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        """Encode ``tokens`` into an array of embeddings."""
        tokens_list = list(tokens)
        matrix = np.zeros((len(tokens_list), self.embedding_dim), dtype="float32")
        for i, token in enumerate(tokens_list):
            matrix[i] = self.get_vector(token)
        return matrix

    def save(self, path: Path) -> None:
        """Persist embeddings to ``path`` using NumPy's NPZ format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self._vectors, vocab=list(self.vocab.keys()))
        LOGGER.info("Saved embeddings to %s", path)

    @classmethod
    def load_from_cache(cls, path: Path) -> "GloVeEmbeddings":
        data = np.load(path, allow_pickle=True)
        vocab_list = data["vocab"].tolist()
        vectors = data["vectors"]
        instance = cls(embedding_dim=vectors.shape[1])
        instance.vocab = {word: index for index, word in enumerate(vocab_list)}
        instance._vectors = vectors
        return instance

    def _oov_vector(self, token: str) -> np.ndarray:
        seed = deterministic_hash(token) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.1, self.embedding_dim).astype("float32")
