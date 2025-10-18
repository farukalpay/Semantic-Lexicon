from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from semantic_lexicon.runtime import run


def test_run_handles_literal_request(tmp_path: Path) -> None:
    result = run("Return only the number 7, nothing else.", workspace=tmp_path)
    assert result.strip() == "7"


def test_run_handles_multiline_json_literal(tmp_path: Path) -> None:
    prompt = textwrap.dedent(
        """
        Return ONLY this JSON:
        {
          "outer": {"inner": [1, 2, {"flag": true}]},
          "note": "brace } inside string"
        }
        """
    )

    result = run(prompt, workspace=tmp_path)
    assert result == '{"outer":{"inner":[1,2,{"flag":true}]},"note":"brace } inside string"}'


def test_run_json_literal_ignores_trailing_text(tmp_path: Path) -> None:
    prompt = 'Return only this JSON: {"ok": true}\nThanks!'

    result = run(prompt, workspace=tmp_path)
    assert result == '{"ok":true}'


@pytest.mark.parametrize("persona", [None, "tutor", "analyst"])
def test_run_falls_back_without_artifacts(tmp_path: Path, persona: str | None) -> None:
    workspace = tmp_path / "artifacts"
    workspace.mkdir()
    response = run("Describe the training pipeline.", persona=persona, workspace=workspace)
    assert isinstance(response, str)
    assert response
