"""The relation of an edge, at the resolution the supply measures need.

Graph construction assigns one of seven structural relations (see
``EdgeWeightConfig``), and two of them cover most of the document: ``text`` is
both heading->paragraph and paragraph->subtext, and ``list_item`` is a bullet
whether a heading or a paragraph anchors it. Those are different relations for
the purpose of supply -- a heading and a paragraph do not deliver the same kind
of context -- so they are subdivided here.

The subdivision is derived, not stored: everything it needs is already on
``Unit``. That matters for the auto-merge ranking, which scores *virtual* edges
from a child to each of its ancestors; those edges never existed during graph
construction and so have no entry in ``ctx.relations``. Only the three relations
that cannot be recovered from the units at all -- ``reference``,
``unreferenced_media`` and ``root``, which are facts about how the edge was
created rather than about the two nodes -- are read from there.
"""

from __future__ import annotations

from .context import (
    KIND_IMAGE,
    KIND_TABLE,
    RELATION_REFERENCE,
    RELATION_ROOT,
    RELATION_UNREFERENCED_MEDIA,
    Edge,
    EdgeContext,
)

# Every relation the supply measures distinguish, in the order the report matrix
# lists them. The first five are the subdivision of the tree relations; the last
# three are carried over from graph construction unchanged.
SUPPLY_RELATIONS = (
    "section",             # heading -> subheading
    "heading_text",        # heading -> paragraph
    "text_text",           # paragraph -> paragraph, subtext, footnote
    "heading_list",        # heading -> bullet
    "text_list",           # anchor paragraph -> bullet
    "media_table",         # text -> table
    "media_figure",        # text -> figure
    "unreferenced_media",  # fallback heading -> media with no caption match
    "reference",           # mentioning text -> media, outside the tree
    "root",                # synthetic document root -> otherwise disconnected node
)

_FROM_CONSTRUCTION = (RELATION_REFERENCE, RELATION_UNREFERENCED_MEDIA, RELATION_ROOT)


def supply_relation(ctx: EdgeContext, edge: Edge) -> str:
    """Which of :data:`SUPPLY_RELATIONS` this edge realizes."""
    relation = ctx.relation(edge)
    if relation in _FROM_CONSTRUCTION:
        return relation
    parent, child = ctx.unit(edge[0]), ctx.unit(edge[1])
    if child.kind == KIND_TABLE:
        return "media_table"
    if child.kind == KIND_IMAGE:
        return "media_figure"
    if parent.is_heading and child.is_heading:
        return "section"
    if child.is_grouped:
        return "heading_list" if parent.is_heading else "text_list"
    return "heading_text" if parent.is_heading else "text_text"


def group_by_relation(ctx: EdgeContext) -> dict[str, list[Edge]]:
    """The document's edges, bucketed by supply relation."""
    buckets: dict[str, list[Edge]] = {}
    for edge in ctx.edges:
        buckets.setdefault(supply_relation(ctx, edge), []).append(edge)
    return buckets
