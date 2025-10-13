# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Text utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable

_WHITESPACE_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lower-case and collapse whitespace in ``text``."""

    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Basic whitespace tokenizer."""

    return normalise_text(text).split()


def build_vocabulary(corpus: Iterable[str]) -> list[str]:
    """Create a sorted vocabulary from ``corpus``."""

    vocab = {token for text in corpus for token in tokenize(text)}
    return sorted(vocab)
