# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from .graph_api import Entity, EntityId, Evidence, GraphAPI

if TYPE_CHECKING:
    from collections.abc import Iterable


def _norm(text: str) -> str:
    return " ".join(text.strip().split()).lower()


class InMemoryGraph(GraphAPI):
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._entities: dict[EntityId, Entity] = {}
        self._alias_to_id: dict[str, EntityId] = {}
        self._spo: dict[tuple[EntityId, str], list[EntityId]] = defaultdict(list)
        self._osp: dict[tuple[str, EntityId], list[EntityId]] = defaultdict(list)
        self._facts: dict[tuple[EntityId, str, EntityId], tuple[Evidence, ...]] = {}
        self._label_cache: dict[EntityId, str] = {}

    def upsert_entity(
        self,
        label: str,
        *,
        entity_id: Optional[EntityId] = None,
        aliases: Optional[Iterable[str]] = None,
        attrs: Optional[dict[str, str]] = None,
    ) -> EntityId:
        eid = entity_id or str(uuid.uuid4())
        if eid in self._entities:
            entity = self._entities[eid]
            entity.label = label or entity.label
            if aliases:
                for alias in aliases:
                    self.add_alias(eid, alias)
            if attrs:
                entity.attrs.update(attrs)
        else:
            entity = Entity(id=eid, label=label, aliases=[], attrs=dict(attrs or {}))
            self._entities[eid] = entity
            self.add_alias(eid, label)
            for alias in aliases or []:
                self.add_alias(eid, alias)
        self._label_cache[eid] = entity.label
        return eid

    def add_alias(self, entity_id: EntityId, alias: str) -> None:
        entity = self._entities[entity_id]
        entity.aliases.append(alias)
        self._alias_to_id[_norm(alias)] = entity_id

    def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def find_entity_by_surface(self, text: str) -> Optional[EntityId]:
        return self._alias_to_id.get(_norm(text))

    def upsert_relation(
        self,
        s: EntityId,
        relation: str,
        o: EntityId,
        evidence: Optional[Iterable[Evidence]] = None,
    ) -> None:
        key = (s, relation, o)
        if key not in self._facts:
            self._facts[key] = tuple(evidence or ())
            self._spo[(s, relation)].append(o)
            self._osp[(relation, o)].append(s)

    def objects(self, s: EntityId, relation: str) -> list[EntityId]:
        return self._spo.get((s, relation), [])

    def subjects(self, relation: str, o: EntityId) -> list[EntityId]:
        return self._osp.get((relation, o), [])

    def has_fact(self, s: EntityId, relation: str, o: EntityId) -> bool:
        return (s, relation, o) in self._facts

    def neighbors(self, entity_id: EntityId) -> list[EntityId]:
        out: set[EntityId] = set()
        for (s, _relation), objects in self._spo.items():
            if s == entity_id:
                out.update(objects)
        for (_relation, o), subjects in self._osp.items():
            if o == entity_id:
                out.update(subjects)
        return list(out)

    def label(self, entity_id: EntityId) -> str:
        if entity_id in self._label_cache:
            return self._label_cache[entity_id]
        entity = self._entities[entity_id]
        self._label_cache[entity_id] = entity.label
        return entity.label

    def all_entities(self) -> Iterable[EntityId]:
        return self._entities.keys()

    def save(self, path: str) -> None:
        payload = {
            "entities": [
                {
                    "id": entity.id,
                    "label": entity.label,
                    "aliases": entity.aliases,
                    "attrs": entity.attrs,
                }
                for entity in self._entities.values()
            ],
            "facts": [
                {
                    "s": s,
                    "r": r,
                    "o": o,
                    "evidence": [evidence.__dict__ for evidence in self._facts[(s, r, o)]],
                }
                for (s, r, o) in self._facts.keys()
            ],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self._reset()
        for entity_payload in payload["entities"]:
            self.upsert_entity(
                entity_payload["label"],
                entity_id=entity_payload["id"],
                aliases=entity_payload.get("aliases") or [],
                attrs=entity_payload.get("attrs") or {},
            )
        for fact in payload["facts"]:
            evidence = tuple(Evidence(**ev) for ev in fact.get("evidence", []))
            self.upsert_relation(fact["s"], fact["r"], fact["o"], evidence)
