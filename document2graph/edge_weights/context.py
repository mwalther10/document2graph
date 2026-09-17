"""The input every edge weight metric reads from.

Metrics used to take a ``{node_id: text}`` mapping, which is enough for
similarity but not for anything structural or linguistic. ``EdgeContext``
carries the units, the edges, and the structural relation of each edge, so a
metric can ask what kind of unit a node is, what its parent looks like, and
which other parent in the document to use as a control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

Edge = tuple[str, str]
WeightedEdge = tuple[str, str, float]

# Structural relations, mirroring the per-relation weights on EdgeWeightConfig.
RELATION_SECTION = "section"
RELATION_TEXT = "text"
RELATION_LIST_ITEM = "list_item"
RELATION_MEDIA = "media"
RELATION_UNREFERENCED_MEDIA = "unreferenced_media"
RELATION_REFERENCE = "reference"
RELATION_ROOT = "root"

KIND_TEXT = "text"
KIND_IMAGE = "image"
KIND_TABLE = "table"
KIND_ROOT = "root"


@dataclass(frozen=True)
class Unit:
    """One node of the document tree, reduced to what a weight metric needs."""

    node_id: str
    text: str
    # docling's own item label: text, list_item, section_header, caption, footnote, ...
    label: str = ""
    # the pipeline's level label: "Title" | "Heading" | "Root" | "Unreferenced Image" | ""
    level_label: str = ""
    level: int = 0
    sequence_no: int = 0
    is_grouped: bool = False
    kind: str = KIND_TEXT

    @property
    def is_media(self) -> bool:
        return self.kind in (KIND_IMAGE, KIND_TABLE)

    @property
    def is_heading(self) -> bool:
        return self.level_label in ("Title", "Heading", "Root")


@dataclass
class EdgeContext:
    """Units, edges and relations of one document."""

    units: dict[str, Unit]
    edges: list[Edge]
    relations: dict[Edge, str] = field(default_factory=dict)

    def text(self, node_id: str) -> str:
        unit = self.units.get(node_id)
        return unit.text if unit else ""

    def unit(self, node_id: str) -> Unit:
        return self.units.get(node_id) or Unit(node_id=node_id, text="")

    def relation(self, edge: Edge) -> str:
        return self.relations.get(edge, RELATION_TEXT)

    @property
    def texts(self) -> dict[str, str]:
        return {node_id: unit.text for node_id, unit in self.units.items()}

    @cached_property
    def parents_of(self) -> dict[str, list[str]]:
        parents: dict[str, list[str]] = {}
        for parent_id, child_id in self.edges:
            parents.setdefault(child_id, []).append(parent_id)
        return parents

    @cached_property
    def _parent_ids(self) -> list[str]:
        """Every node that parents at least one edge, in a stable order."""
        seen = {parent_id for parent_id, _ in self.edges}
        return sorted(seen)

    def ancestors_of(self, node_id: str) -> list[str]:
        """Every node reachable upward from ``node_id``, nearest first.

        The candidates the auto-merge ranking scores a child against, and the
        set a counterfactual control has to stay out of.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        frontier = list(self.parents_of.get(node_id, []))
        while frontier:
            current = frontier.pop(0)
            if current in seen or current == node_id:
                continue
            seen.add(current)
            ordered.append(current)
            frontier.extend(self.parents_of.get(current, []))
        return ordered

    def with_edges(self, edges: list[Edge]) -> "EdgeContext":
        """The same document, scored over a different edge set.

        Counterfactual controls and the virtual child->ancestor edges of the
        merge ranking are both edges that never existed during graph
        construction; sharing the units means the term statistics and the texts
        stay identical, so only the pairing changes.
        """
        return EdgeContext(units=self.units, edges=list(edges), relations=dict(self.relations))

    def is_ancestor(self, candidate: str, node_id: str) -> bool:
        """Whether ``candidate`` reaches ``node_id`` by following parent edges upward."""
        seen: set[str] = set()
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            if current == candidate and current != node_id:
                return True
            frontier.extend(self.parents_of.get(current, []))
        return candidate in seen and candidate != node_id

    def counterfactual_parent(self, edge: Edge) -> str | None:
        """A parent from this document that is *not* the child's own ancestor.

        The control for conditional surprisal: conditioning the child on this
        instead of its real parent should collapse a genuine dependency while
        leaving shared-domain vocabulary effects intact. Chosen deterministically
        by walking the parent list from a hash-free offset, so a rerun of the
        same document picks the same control.
        """
        parent_id, child_id = edge
        candidates = self._parent_ids
        if not candidates:
            return None
        try:
            start = candidates.index(parent_id)
        except ValueError:
            start = 0
        # step half the list away, then walk until a usable control turns up
        offset = max(1, len(candidates) // 2)
        for step in range(len(candidates)):
            candidate = candidates[(start + offset + step) % len(candidates)]
            if candidate in (parent_id, child_id):
                continue
            if not self.text(candidate).strip():
                continue
            if self.is_ancestor(candidate, child_id):
                continue
            return candidate
        return None

    @classmethod
    def from_texts(cls, edges: list[Edge], texts: dict[str, str],
                   relations: dict[Edge, str] | None = None) -> "EdgeContext":
        """Build a context from bare texts, for metrics that need nothing else."""
        units = {node_id: Unit(node_id=node_id, text=text) for node_id, text in texts.items()}
        for parent_id, child_id in edges:
            for node_id in (parent_id, child_id):
                units.setdefault(node_id, Unit(node_id=node_id, text=""))
        return cls(units=units, edges=list(edges), relations=dict(relations or {}))
