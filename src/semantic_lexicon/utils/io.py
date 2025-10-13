# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""I/O utilities."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path


def _open_text(path: Path) -> Iterator[str]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, mode="rt", encoding="utf8") as handle:
            yield from handle
    else:
        with path.open("r", encoding="utf8") as handle:
            yield from handle


def read_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    """Stream JSON lines from ``path``.

    Supports plain-text ``.jsonl`` files as well as gzip-compressed ``.jsonl.gz`` files.
    """

    for line in _open_text(Path(path)):
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
