from docling_core.types.doc.document import TextItem
from pydantic import BaseModel

class TextSnippet(BaseModel):
    text_item: TextItem
    line_heights: list[float] = []