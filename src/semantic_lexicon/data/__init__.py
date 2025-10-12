"""Bundled data for the Semantic Lexicon package."""

from __future__ import annotations

import json
from importlib import resources
from typing import List, Dict


def load_sample_corpus() -> List[Dict[str, str]]:
    with resources.files(__package__).joinpath("sample_corpus.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


__all__ = ["load_sample_corpus"]
