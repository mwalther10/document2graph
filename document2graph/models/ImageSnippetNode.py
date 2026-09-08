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
    referencing_node_ids: list[str] = []  # snippet_ids of all text nodes that mention this media item
    bbox: BoundingBox
    page_no: int
    # every location this node occupies in the PDF; bbox/page_no above are the first
    # of them. A table continued on the next page has several.
    provenance: list[Provenance] = []