from docling_core.types.doc.document import TextItem
from pydantic import BaseModel

# Where a snippet sits on the page. Only "body" text forms the document outline:
# the masthead of the first page, the text of a boxed sidebar and the labels inside a
# figure are all set as headings by docling but are not sections of the document.
REGION_BODY = "body"
REGION_FRONT_MATTER = "front_matter"
REGION_SIDEBAR = "sidebar"
REGION_FIGURE = "figure"

class TextSnippet(BaseModel):
    text_item: TextItem
    line_heights: list[float] = []
    font_key: str | None = None
    region: str = REGION_BODY
