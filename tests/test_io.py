from __future__ import annotations

import gzip
import json
from pathlib import Path

from semantic_lexicon.utils.io import read_jsonl


def test_read_jsonl_supports_gzip(tmp_path: Path) -> None:
    records = [{"text": "alpha"}, {"text": "beta"}]
    gz_path = tmp_path / "records.jsonl.gz"

    with gzip.open(gz_path, mode="wt", encoding="utf8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    parsed = list(read_jsonl(gz_path))

    assert parsed == records
