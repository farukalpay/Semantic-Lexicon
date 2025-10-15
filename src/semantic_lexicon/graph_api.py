# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

EntityId = str


@dataclass(frozen=True)
class Evidence:
    source: str
    confidence: float = 1.0
    note: str = ""


@dataclass
class Entity:
    id: EntityId
    label: str
    aliases: list[str] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    subject: EntityId
    relation: str
    object: EntityId
    evidence: tuple[Evidence, ...] = ()


class GraphAPI(Protocol):
    """Minimal graph contract your oracles/decoder rely on."""

    def upsert_entity(
        self,
        label: str,
        *,
        entity_id: Optional[EntityId] = None,
        aliases: Optional[Iterable[str]] = None,
        attrs: Optional[dict[str, str]] = None,
    ) -> EntityId: ...

    def add_alias(self, entity_id: EntityId, alias: str) -> None: ...

    def get_entity(self, entity_id: EntityId) -> Optional[Entity]: ...

    def find_entity_by_surface(self, text: str) -> Optional[EntityId]: ...

    def upsert_relation(
        self,
        s: EntityId,
        relation: str,
        o: EntityId,
        evidence: Optional[Iterable[Evidence]] = None,
    ) -> None: ...

    def objects(self, s: EntityId, relation: str) -> list[EntityId]: ...

    def subjects(self, relation: str, o: EntityId) -> list[EntityId]: ...

    def has_fact(self, s: EntityId, relation: str, o: EntityId) -> bool: ...

    def neighbors(self, entity_id: EntityId) -> list[EntityId]: ...

    def label(self, entity_id: EntityId) -> str: ...

    def all_entities(self) -> Iterable[EntityId]: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...
