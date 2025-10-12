from __future__ import annotations

from pathlib import Path

import sys
from typing import Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_lexicon.config import SemanticModelConfig


@pytest.fixture
def config() -> SemanticModelConfig:
    return SemanticModelConfig()


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
