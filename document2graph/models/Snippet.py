from pydantic import BaseModel
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.base import BoundingBox

from .Provenance import Provenance
from .TextSnippet import REGION_BODY

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
    # where the snippet sits on the page: body | front_matter | sidebar | figure
    region: str = REGION_BODY
    page_no: int
    bbox: BoundingBox
    charspan: tuple[int, int] | None = None
    provenance: list[Provenance] = []
    text: str
    docling_parent_ref: RefItem | None
    docling_self_ref: RefItem | None