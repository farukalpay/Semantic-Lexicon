from __future__ import annotations

from pathlib import Path

import pytest

from semantic_lexicon.runtime import run


def test_run_handles_literal_request(tmp_path: Path) -> None:
    result = run("Return only the number 7, nothing else.", workspace=tmp_path)
    assert result.strip() == "7"


@pytest.mark.parametrize("persona", [None, "tutor", "analyst"])
def test_run_falls_back_without_artifacts(tmp_path: Path, persona: str | None) -> None:
    workspace = tmp_path / "artifacts"
    workspace.mkdir()
    response = run("Describe the training pipeline.", persona=persona, workspace=workspace)
    assert isinstance(response, str)
    assert response
