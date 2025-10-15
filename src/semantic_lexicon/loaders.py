# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import csv
import json
from typing import Dict, Iterable

from .graph_api import Evidence
from .graph_memory import InMemoryGraph


def load_triples_csv(
    graph: InMemoryGraph,
    path: str,
    *,
    subject_col: str = "subject",
    relation_col: str = "relation",
    object_col: str = "object",
    source_col: str | None = None,
) -> None:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject = row[subject_col]
            obj = row[object_col]
            subject_id = graph.find_entity_by_surface(subject) or graph.upsert_entity(subject)
            object_id = graph.find_entity_by_surface(obj) or graph.upsert_entity(obj)
            evidence: Iterable[Evidence] = []
            if source_col and row.get(source_col):
                evidence = [Evidence(source=row[source_col])]
            graph.upsert_relation(subject_id, row[relation_col], object_id, evidence)


def load_entities_jsonl(
    graph: InMemoryGraph,
    path: str,
    *,
    label_key: str = "label",
    aliases_key: str = "aliases",
    attrs_key: str = "attrs",
) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            payload: Dict[str, object] = json.loads(line)
            graph.upsert_entity(
                str(payload[label_key]),
                aliases=[str(alias) for alias in payload.get(aliases_key, [])],
                attrs={str(k): str(v) for k, v in (payload.get(attrs_key) or {}).items()},
            )
