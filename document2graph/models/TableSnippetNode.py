
from .ImageSnippetNode import ImageSnippetNode


class TableSnippetNode(ImageSnippetNode):
    markdown_serialization: str
    html_serialization: str
    # docling refs of the table items merged into this one as continuations of it;
    # empty for a table that was not broken across a page break
    continuation_refs: list[str] = []

    def content_text(self) -> str:
        """Caption above the table body.

        The caption is the strongest retrieval signal a table has -- it is the only
        part of it written as a sentence -- so it belongs in the node's text and not
        only in its own field.
        """
        return "\n".join(part for part in (self.caption_text, self.markdown_serialization)
                          if part).strip()
