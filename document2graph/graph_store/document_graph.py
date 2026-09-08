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
from ..models.Provenance import Provenance

if TYPE_CHECKING:  # avoid a circular import: the extractor imports the store
    from ..document2graph_extractor.snippet_graph_constructor import SnippetGraph

SCHEMA_VERSION = 1

# snippet_type of the synthetic root node materialized for documents with no title node
ROOT_SNIPPET_TYPE = "root"


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
        )
