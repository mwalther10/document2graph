from pydantic import BaseModel
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.base import BoundingBox

class Snippet(BaseModel):
    snippet_id: str
    type: str
    document_id: str
    sequence_no: int
    label: str
    level: int
    level_label: str 
    parent_id: str | None
    is_grouped: bool = False
    page_no: int
    bbox: BoundingBox
    text: str
    docling_parent_ref: RefItem | None
    docling_self_ref: RefItem | None