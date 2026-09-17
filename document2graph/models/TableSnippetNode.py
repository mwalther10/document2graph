
from .ImageSnippetNode import ImageSnippetNode


class TableSnippetNode(ImageSnippetNode):
    markdown_serialization: str
    html_serialization: str
    # docling refs of the table items merged into this one as continuations of it;
    # empty for a table that was not broken across a page break
    continuation_refs: list[str] = []
