from pydantic import BaseModel
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.base import BoundingBox

from .TextSnippet import REGION_BODY

class TextSnippetNode(BaseModel):
    snippet_id: str
    document_id: str
    label: str
    sequence_no: int
    level: int
    level_label: str 
    parent_id: str | None
    docling_parent_ref: RefItem 
    docling_self_ref: RefItem 
    is_grouped: bool = False
    text: str
    bbox: BoundingBox
    charspan: tuple[int, int]
    page_no: int
    line_heights: list[float] = []
    font_key: str | None = None
    level_height: float | None = None
    # where the snippet sits on the page: body | front_matter | sidebar | figure
    region: str = REGION_BODY