from pydantic import BaseModel
from docling_core.types.doc.base import BoundingBox


class Provenance(BaseModel):
    """Where a unit of a document sits in the source PDF.

    Every retrieval unit this package emits -- a graph node, a ``Snippet``, a
    ``GraphSnippet`` or a baseline ``Chunk`` -- carries a ``provenance`` list of
    these, so a consumer aligning graph nodes against baseline chunks has one
    code path instead of one per representation.

    A unit has more than one entry whenever it spans more than one region of the
    PDF: a paragraph stitched back together across a column or page break, a
    table continued on the next page, or a chunk merged from several document
    items. ``bbox`` is in the coordinate origin docling reported it in.
    """

    page_no: int
    bbox: BoundingBox | None = None
    charspan: tuple[int, int] | None = None
