from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import DoclingDocument, TableItem, TextItem, PictureItem
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import GroupItem, RefItem
import os
import pickle
from .ids import clean_filename
from .log import Log


def docling_doc_path(cache_dir: str, source: str) -> str:
    """Where the parsed docling document of ``source`` is cached."""
    return os.path.join(cache_dir, f"{clean_filename(source)}_docling_doc.json")


class Extractor:
    """Parses a PDF with docling and serializes it for the graph builders.

    The parse is by far the most expensive step, so the parsed document is
    cached as json under ``cache_dir`` and reused: pass ``cache_dir`` to reuse
    it, and ``refresh=True`` to re-parse and overwrite. Everything downstream of
    this class works off the resulting ``DoclingDocument``, so ablations of
    post-parse behaviour never need the PDF re-parsed.
    """

    def __init__(self, source: str, pipeline_options: PdfPipelineOptions | None = None,
                 cache_dir: str | None = None, refresh: bool = False):
        self.source = source
        self.logger = Log("Extractor").logger
        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options or PdfPipelineOptions())
        })
        self.cache_dir = cache_dir
        self.doc = self._load_or_convert(refresh)

    def _load_or_convert(self, refresh: bool) -> DoclingDocument:
        cache_file = docling_doc_path(self.cache_dir, self.source) if self.cache_dir else None
        if cache_file and not refresh and os.path.isfile(cache_file):
            self.logger.info(f"Loading cached docling document from {cache_file}")
            return DoclingDocument.load_from_json(cache_file)

        self.logger.info(f"Converting {self.source} with docling (slow)...")
        doc = self.converter.convert(self.source).document
        if cache_file:
            os.makedirs(self.cache_dir, exist_ok=True)
            doc.save_as_json(cache_file)
        return doc


    def is_furniture(self, item):
        """
        Check if the item is furniture (e.g., title page, table of contents).
        """
        return item.content_layer.value == "furniture"

    def is_text(self, item):
        """
        Check if the item is a text item.
        """
        return isinstance(item, TextItem)

    def is_picture(self, item):
        """
        Check if the item is a picture item.
        """
        return isinstance(item, PictureItem)

    def is_table(self, item):
        """
        Check if the item is a table item.
        """
        return isinstance(item, TableItem)

    def is_group(self, item):
        """
        Check if the item is a group item.
        """
        return isinstance(item, GroupItem)

    def serialize_ref_items(
        self, ref_items: list[RefItem], doc: DoclingDocument
    ) -> tuple[list[str], list[str]]:
        """
        Serialize the reference items to a string representation.
        This function is necessary to handle also string representations of groups, tables and images.
        Only for annotation of extracted doc necessary.
        """
        serialized_items = []
        serialized_labels = []
        for refItem in ref_items:
            item = refItem.resolve(doc)
            if self.is_furniture(item):
                continue
            if self.is_text(item):
                serialized_items.append(item.text)
                serialized_labels.append(item.label.value)
            elif self.is_picture(item):
                serialized_items.append(item.caption_text(doc))
                serialized_labels.append(item.label.value)
            elif self.is_table(item):
                serialized_items.append(item.export_to_markdown(doc))
                serialized_labels.append(item.label.value)
            elif self.is_group(item):
                # Recursively serialize group items
                group_items, group_labels = self.serialize_ref_items(item.children, doc)
                serialized_items.append(group_items)
                serialized_labels.append(group_labels)
            else:
                serialized_items.append(item.text)
                serialized_labels.append(item.label.value)
        return serialized_items, serialized_labels


    def serialize_doc(self, doc: DoclingDocument) -> dict:
        """
        Serialize the document to a string representation.
        Leave furniture out. Bring text in order, to see if we can manually label the paragraphs.
        """
        full_text, full_text_labels = self.serialize_ref_items(
            doc.body.children, doc
        )
        return {
            "full_text": full_text,
            "full_text_labels": full_text_labels,
            "pictures": doc.pictures,
            "tables": doc.tables,
            "texts": doc.texts,
        }

    def extract(self, save_dir: str, filename: str) -> dict:
        """
        Extract and serialize the document, saving intermediate results.
        """
        os.makedirs(save_dir, exist_ok=True)

        doc_json = f"{save_dir}/{filename}_docling_doc.json"
        # the cache holds this file already when the document came from it
        if not os.path.isfile(doc_json):
            self.logger.info(f"Saving parsed pdf as {filename}.json")
            self.doc.save_as_json(doc_json)
        self.logger.info(f"Start extracting {filename}...")
        result = self.serialize_doc(self.doc)
        with open(f"{save_dir}/{filename}_serialized_doc.pkl", "wb") as f:
            pickle.dump(result, f)
        return result
