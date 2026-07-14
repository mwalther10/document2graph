import uuid

from docling_core.types.doc.document import TextItem, SectionHeaderItem, RefItem

from ..models.TextSnippet import TextSnippet
from ..models.Document import Document
from ..models.DocumentMetadata import DocumentMetadata, MetadataExtractionConfig
from ..utils.log import Log


class DocumentMetadataExtractor:
    """Rule-based extraction of document metadata (title, authors, institutions, ...)
    from the text snippets of a parsed document."""

    def __init__(self, text_items: list[TextSnippet]):
        self.text_items = text_items
        self.logger = Log("DocumentMetadataExtractor").logger

    def _safe_check_substring(self, search_string: str, item: TextItem) -> bool:
        try:
            return search_string.lower() in item.text.lower()
        except Exception as e:
            self.logger.warning(f"Error checking item text: {e}")
            return False

    def _match_metadata_by_section_header(self, section_header: SectionHeaderItem | TextItem, texts: list[TextItem]) -> RefItem | None:
        bottom = section_header.prov[0].bbox.b
        left = section_header.prov[0].bbox.l

        for text in texts:
            text_top = text.prov[0].bbox.t
            text_left = text.prov[0].bbox.l
            if (abs(bottom - text_top) < 6) and (abs(left - text_left) < 6) and text.parent is not None and section_header.parent is not None:
                if text.parent.get_ref() == section_header.parent.get_ref():
                    return text.get_ref()
                if text.parent.get_ref() != section_header.parent.get_ref():
                    return text.parent.get_ref()
        return None

    def extract_title(self, title_page_snippets: list[TextSnippet]) -> str:
        # condition: on the configured title page, largest font size, and on top of page
        title = max(title_page_snippets, key=lambda s: s.line_heights[0] if s.line_heights else 0)
        if not title.line_heights:
            return title.text_item.text
        for snippet in title_page_snippets:
            if snippet.line_heights and snippet.line_heights[0] >= title.line_heights[0] and snippet.text_item.prov[0].bbox.t < title.text_item.prov[0].bbox.t:
                title = snippet
        return title.text_item.text

    def extract(self, filename: str, document_type: str, config: MetadataExtractionConfig) -> Document:
        metadata_values: dict[str, str] = {}
        for field_name in ["version", "authors", "institutions", "bibliography", "correspondence"]:
            field_cfg = getattr(config, field_name)
            if field_cfg is None:
                metadata_values[field_name] = ""
                continue
            page_items = [item.text_item for item in self.text_items
                          if field_cfg.pages[0] <= item.text_item.prov[0].page_no <= field_cfg.pages[1]]
            matched_ref = None
            for text_item in page_items:
                if self._safe_check_substring(field_cfg.label, text_item):
                    matched_ref = self._match_metadata_by_section_header(text_item, page_items)
                    if matched_ref is None:
                        matched_ref = text_item.get_ref()
                    break
            metadata_values[field_name] = matched_ref.cref if matched_ref else ""

        title_page_snippets = [item for item in self.text_items
                               if item.text_item.prov[0].page_no == config.title_page]
        title = self.extract_title(title_page_snippets) if title_page_snippets else ""

        return Document(
            document_id=str(uuid.uuid4()),
            title=title,
            document_type=document_type,
            filename=filename,
            metadata=DocumentMetadata(**metadata_values),
        )
