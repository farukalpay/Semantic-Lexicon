"""I/O helpers for reading and writing structured data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import yaml


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield objects from a JSON Lines file."""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Write ``data`` to ``path`` as JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=indent, ensure_ascii=False)
        stream.write("\n")


def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    """Load a mapping from YAML or JSON depending on the file suffix."""
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(stream) or {}
        return json.load(stream)
