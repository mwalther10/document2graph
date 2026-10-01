from pydantic import BaseModel
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.base import BoundingBox
from .Provenance import Provenance
from .TextSnippetNode import TextSnippetNode


class ImageSnippetNode(BaseModel):
    snippet_id: str
    document_id: str
    label: str
    sequence_no: int
    level: int
    level_label: str 
    parent_id: str | None = None
    docling_parent_ref: RefItem | None
    docling_self_ref: RefItem | None
    is_grouped: bool = False
    caption_nodes: list[TextSnippetNode] | list = []
    caption_text: str
    # the text printed inside the figure, in reading order: what absorb_figure_text
    # folded in. Empty for a picture with nothing written in it, and for tables,
    # whose contents are the markdown serialization instead.
    figure_text: str = ""
    referencing_node_ids: list[str] = []  # snippet_ids of all text nodes that mention this media item
    bbox: BoundingBox
    page_no: int
    # every location this node occupies in the PDF; bbox/page_no above are the first
    # of them. A table continued on the next page has several.
    provenance: list[Provenance] = []

    def content_text(self) -> str:
        """Everything this node reads as, caption first.

        A media node has no single ``text`` field the way a text node does -- what it
        is made of depends on what it is -- so every consumer that wants "the text of
        this node" asks here rather than picking a field, which is how the persisted
        graph came to hold tables without their captions.
        """
        return "\n".join(part for part in (self.caption_text, self.figure_text) if part).strip()