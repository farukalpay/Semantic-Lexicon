# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import csv
import json
from typing import Any

from .graph_api import Evidence, GraphAPI


def load_triples_csv(
    graph: GraphAPI,
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
            evidence: list[Evidence] = []
            if source_col and row.get(source_col):
                evidence = [Evidence(source=row[source_col])]
            graph.upsert_relation(subject_id, row[relation_col], object_id, evidence)


def load_entities_jsonl(
    graph: GraphAPI,
    path: str,
    *,
    label_key: str = "label",
    aliases_key: str = "aliases",
    attrs_key: str = "attrs",
) -> None:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            if not isinstance(data, dict):
                msg = "Each JSONL line must decode to an object."
                raise ValueError(msg)

            payload: dict[str, Any] = data
            label = str(payload[label_key])

            aliases_raw = payload.get(aliases_key, [])
            if isinstance(aliases_raw, list):
                aliases = [str(alias) for alias in aliases_raw]
            else:
                aliases = [str(aliases_raw)] if aliases_raw else []

            attrs_raw = payload.get(attrs_key) or {}
            if isinstance(attrs_raw, dict):
                attrs = {str(key): str(value) for key, value in attrs_raw.items()}
            else:
                attrs = {}

            graph.upsert_entity(label, aliases=aliases, attrs=attrs)
