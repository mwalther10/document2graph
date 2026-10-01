"""One retrievable unit: a node, plus whatever context was merged into it."""

from pydantic import BaseModel, Field

from .Provenance import Provenance


class MergedEdge(BaseModel):
    """One edge a merge decision followed, and what the decision rested on.

    Recorded so a condition can be audited rather than only scored: which edges a
    policy actually merged, at what weight, is the difference between "the metric
    helped" and "the threshold happened to land somewhere useful".
    """

    parent_id: str
    child_id: str
    weight: float
    # product of the weights from the unit's own node up to and including this edge
    cumulative: float


class MergedUnit(BaseModel):
    """A unit of retrieval, built around one node of the document graph.

    ``text`` reads outermost context first and the unit's own node last, which is
    the order a reader meets them in: heading, then paragraph, then the bullet.

    ``breadcrumb`` is kept *out* of ``text`` on purpose. Prepending the heading
    path is its own condition in a retrieval comparison, and one that any
    segmentation can be given, so a merge policy that folded it in silently would
    make "structure-aware" and "heading-prepending" impossible to tell apart.
    """

    unit_id: str          # snippet_id of the node this unit is built around
    document_id: str
    text: str
    # every snippet contributing text, in the order it appears in `text`; the
    # unit's own node is always the last entry
    member_ids: list[str] = Field(default_factory=list)
    # texts of the heading ancestors, outermost first
    breadcrumb: list[str] = Field(default_factory=list)
    token_count: int = 0
    # the locations of every member, so a merged unit can still be placed on the page
    provenance: list[Provenance] = Field(default_factory=list)
    page_no: int = 0
    node_type: str = "text"
    # the edges the policy merged along, nearest ancestor first; empty when the
    # unit is the bare node
    merged_edges: list[MergedEdge] = Field(default_factory=list)

    @property
    def is_merged(self) -> bool:
        return len(self.member_ids) > 1

    @property
    def pages(self) -> list[int]:
        """Every page this unit draws text from, in order."""
        seen: list[int] = []
        for entry in self.provenance:
            if entry.page_no not in seen:
                seen.append(entry.page_no)
        return seen
