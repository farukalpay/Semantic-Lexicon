"""Text processing helpers used throughout the Semantic Lexicon package."""

from __future__ import annotations

import re
from typing import Iterable, List

_WORD_RE = re.compile(r"[\w']+")


def normalise_text(value: str) -> str:
    """Normalise text by lowercasing and collapsing whitespace."""
    collapsed = " ".join(value.strip().split())
    return collapsed.lower()


def simple_tokenize(value: str) -> List[str]:
    """Tokenise text using a simple regex-based word splitter."""
    if not value:
        return []
    return _WORD_RE.findall(normalise_text(value))


def sliding_window(tokens: Iterable[str], size: int) -> Iterable[List[str]]:
    """Yield windows of ``size`` tokens from ``tokens``."""
    buffer: List[str] = []
    for token in tokens:
        buffer.append(token)
        if len(buffer) == size:
            yield list(buffer)
            buffer.pop(0)


def ensure_sentence_ending(text: str) -> str:
    """Ensure the text ends with terminal punctuation for readability."""
    if not text:
        return text
    return text if text.endswith((".", "!", "?")) else f"{text}."
