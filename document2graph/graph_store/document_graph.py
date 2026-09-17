"""The serializable, corpus-wide form of a document graph.

``SnippetGraph`` is the raw output of ``SnippetGraphConstructor``: three lists of
heterogeneous nodes keyed by document-local docling refs (``#/texts/12``). Those
refs collide across documents, so a corpus needs ids that are globally unique --
and a harness that runs many retrieval conditions over the same corpus needs the
graph back from disk without re-running docling.

``DocumentGraph`` is that form: one flat list of ``GraphSnippet``, globally
hashed ids, the document metadata attached, and a lossless superset of the node
fields so a reload needs no re-parse.
"""

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from docling_core.types.doc.base import BoundingBox
from pydantic import BaseModel

from ..models.Document import Document
from ..models.PageGeometry import PageGeometry
from ..models.Provenance import Provenance

if TYPE_CHECKING:  # avoid a circular import: the extractor imports the store
    from ..document2graph_extractor.snippet_graph_constructor import SnippetGraph

# 2: added DocumentGraph.pages. A graph written under version 1 still loads --
# pages defaults to empty -- but nothing can place its boxes on a page, so a
# consumer that needs geometry should re-extract rather than read the default as
# "this document has no pages".
# 3: added DocumentGraph.edge_relations, without which a reloaded graph cannot
# reproduce its own weights: three relations are facts about how an edge was
# created and are not recoverable from the two nodes it joins.
SCHEMA_VERSION = 3

# snippet_type of the synthetic root node materialized for documents with no title node
ROOT_SNIPPET_TYPE = "root"

# Separator for the (parent, child) key of DocumentGraph.edge_relations. A JSON
# object cannot key on a pair, and snippet ids are hex digests, so no id can
# contain this.
EDGE_KEY_SEPARATOR = ">"


def edge_key(parent_id: str, child_id: str) -> str:
    """The ``edge_relations`` key of one edge."""
    return f"{parent_id}{EDGE_KEY_SEPARATOR}{child_id}"


def global_id(document_id: str, local_ref: str) -> str:
    """Snippet ids from document2graph are document-local docling refs
    (e.g. "#/texts/12"); hash them with the document id to get a globally
    unique snippet_id."""
    return hashlib.sha256(f"{document_id}:{local_ref}".encode()).hexdigest()


class GraphSnippet(BaseModel):
    """A node of the document graph layer, aligned with document2graph output.

    snippet_id is globally unique (hash of document_id + local_ref);
    local_ref is the document-local docling reference, e.g. "#/texts/12".
    """

    snippet_id: str
    local_ref: str
    parent_local_ref: str | None = None
    document_id: str
    sequence_no: int
    label: str
    snippet_type: str = "text"
    level: int
    level_label: str
    region: str = "body"
    is_grouped: bool = False
    page_no: int
    bbox: BoundingBox | None = None
    charspan: tuple[int, int] | None = None
    provenance: list[Provenance] = []
    text: str
    line_heights: list[float] = []
    font_key: str | None = None
    level_height: float | None = None
    extracted_at: str

    @property
    def parent_id(self) -> str | None:
        """The globally unique snippet_id of this node's parent."""
        if self.parent_local_ref is None:
            return None
        return global_id(self.document_id, self.parent_local_ref)


class DocumentGraph(BaseModel):
    """The document graph layer of a single document."""

    schema_version: int = SCHEMA_VERSION
    document_id: str
    filename: str
    title: str
    document_type: str = ""
    version: str = ""
    authors: str = ""
    institutions: str = ""
    root_id: str  # globally unique snippet_id of the document root (title node or synthetic root)
    snippets: list[GraphSnippet]
    edges: list[tuple[str, str, float]]  # hierarchy tree: (parent snippet_id, child snippet_id, weight)
    reference_edges: list[tuple[str, str, float]] = []  # mentions: (text snippet_id, media snippet_id, weight)
    # the box of every page, so a reloaded graph can place its own bounding boxes
    # without the PDF; empty on graphs written before schema version 2
    pages: list[PageGeometry] = []
    # structural relation of every edge and reference edge, keyed by edge_key().
    # Empty on graphs written before schema version 3.
    #
    # Stored rather than derived because three of the relations -- reference,
    # unreferenced_media and root -- describe how graph construction created the
    # edge, not what the two nodes are, so nothing downstream can recover them.
    # Losing them is not a small loss: unreferenced_media carries the *highest*
    # measured supply prior (1.00) and would otherwise be read as an ordinary
    # media edge (0.50).
    edge_relations: dict[str, str] = {}

    def relation(self, parent_id: str, child_id: str) -> str | None:
        """The structural relation of one edge, or None if it was not recorded."""
        return self.edge_relations.get(edge_key(parent_id, child_id))

    def page(self, page_no: int) -> PageGeometry | None:
        """The box of one page, or None when the page table was not recorded.

        Returning None rather than raising: a graph written under schema version
        1 has no page table at all, and a consumer should be able to tell that
        apart from a page number that does not exist.
        """
        for page in self.pages:
            if page.page_no == page_no:
                return page
        return None

    @classmethod
    def from_snippet_graph(cls, graph: "SnippetGraph", document: Document) -> "DocumentGraph":
        """Build the globally identified graph from a constructor result."""
        document_id = document.document_id
        extracted_at = datetime.now(timezone.utc).isoformat()

        snippets: list[GraphSnippet] = []
        for node, snippet_type, text_attr in (
            *[(n, "text", "text") for n in graph.text_nodes],
            *[(n, "image", "caption_text") for n in graph.image_nodes],
            *[(n, "table", "markdown_serialization") for n in graph.table_nodes],
        ):
            text = (getattr(node, text_attr, "") or "").replace("\x00", "")
            snippets.append(
                GraphSnippet(
                    snippet_id=global_id(document_id, node.snippet_id),
                    local_ref=node.snippet_id,
                    parent_local_ref=node.parent_id,
                    document_id=document_id,
                    sequence_no=node.sequence_no,
                    label=node.label,
                    snippet_type=snippet_type,
                    level=node.level,
                    level_label=node.level_label,
                    region=getattr(node, "region", "body"),
                    is_grouped=node.is_grouped,
                    page_no=node.page_no,
                    bbox=node.bbox,
                    charspan=getattr(node, "charspan", None),
                    provenance=node.provenance,
                    text=text,
                    line_heights=getattr(node, "line_heights", []),
                    font_key=getattr(node, "font_key", None),
                    level_height=getattr(node, "level_height", None),
                    extracted_at=extracted_at,
                )
            )

        # the root is usually the document title node and already part of the
        # snippets; only the synthetic fallback root needs to be materialized.
        root_local_ref = graph.root_id
        root_id = global_id(document_id, root_local_ref)
        if root_local_ref not in {s.local_ref for s in snippets}:
            snippets.append(
                GraphSnippet(
                    snippet_id=root_id,
                    local_ref=root_local_ref,
                    document_id=document_id,
                    sequence_no=-1,
                    label="document_root",
                    snippet_type=ROOT_SNIPPET_TYPE,
                    level=-1,
                    level_label="Root",
                    page_no=0,
                    text=document.title or document.filename,
                    extracted_at=extracted_at,
                )
            )

        return cls(
            document_id=document_id,
            filename=document.filename,
            title=document.title or document.filename,
            document_type=document.document_type,
            version=document.metadata.version,
            authors=document.metadata.authors,
            institutions=document.metadata.institutions,
            root_id=root_id,
            snippets=snippets,
            edges=[
                (global_id(document_id, parent), global_id(document_id, child), float(weight))
                for parent, child, weight in graph.edges
            ],
            reference_edges=[
                (global_id(document_id, source), global_id(document_id, target), float(weight))
                for source, target, weight in graph.reference_edges
            ],
            pages=list(graph.pages),
            edge_relations={
                edge_key(global_id(document_id, parent), global_id(document_id, child)): relation
                for (parent, child), relation in graph.relations.items()
            },
        )
