"""I/O utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path


def read_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    """Stream JSON lines from ``path``."""

    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Write an iterable of mappings to ``path`` as JSON lines."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
