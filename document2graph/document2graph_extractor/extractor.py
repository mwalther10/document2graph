import os

from .snippet_graph_constructor import SnippetGraphConstructor
from ..utils.base_extractor import Extractor
from tqdm import tqdm
import json
from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument
from ..models.Snippet import Snippet
from ..models.ExtractorConfig import ExtractorConfig
from typing import Any, List
from ..graph_store import DocumentGraph, save_graph
from ..utils.log import Log


class SnippetType:
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"

class DocumentGraphExtractor:
    def __init__(self, config: ExtractorConfig):
        self.pdf_path = config.pdf_path
        self.data_path = config.data_path
        self.pdfPipelineOptions = config.pdfPipelineOptions
        self.save_json = config.save_json
        self.logger = Log("DocumentGraphExtractor").logger
        self.document_type = config.document_type
        self.metadata_config = config.metadata_config
        self.edge_weights = config.edge_weights
        self.flags = config.flags
        self.use_docling_cache = config.use_docling_cache
        self.refresh_docling_cache = config.refresh_docling_cache

    def _run(self, filename: str) -> dict[str, Any]:
        sample = os.path.join(self.pdf_path, filename)
        clean_filename = os.path.splitext(os.path.basename(sample))[0]
        raw_text_save_dir = os.path.join(self.data_path, "raw_texts/")
        graph_save_dir = os.path.join(self.data_path, "nx_graphs/")
        os.makedirs(raw_text_save_dir, exist_ok=True)
        os.makedirs(graph_save_dir, exist_ok=True)

        extractor = Extractor(
            source=sample,
            pipeline_options=self.pdfPipelineOptions,
            cache_dir=raw_text_save_dir if self.use_docling_cache else None,
            refresh=self.refresh_docling_cache,
        )
        extractor.extract(save_dir=raw_text_save_dir, filename=clean_filename)
        parser = DoclingPdfParser()
        pdf_doc: PdfDocument = parser.load(path_or_stream=sample)

        snippet_to_graph = SnippetGraphConstructor(
            pdf_doc,
            extractor.doc,
            clean_filename,
            document_type=self.document_type,
            metadata_config=self.metadata_config,
            edge_weights=self.edge_weights,
            flags=self.flags,
        )
        graph = snippet_to_graph.get_graph(
            save_to=f"{graph_save_dir}/{clean_filename}.gexf"
        )
        doc_metadata = snippet_to_graph.document_metadata
        document_graph = DocumentGraph.from_snippet_graph(graph, doc_metadata)
        if self.save_json:
            # GEXF drops bbox, page_no, charspan and the table serializations; the
            # json is the form that can be loaded back without re-running docling
            save_graph(document_graph, os.path.join(self.data_path, "graphs", f"{clean_filename}_graph.json"))

        return {
            "text_nodes": graph.text_nodes,
            "image_nodes": graph.image_nodes,
            "table_nodes": graph.table_nodes,
            "edges": graph.edges,
            "reference_edges": graph.reference_edges,
            "root_id": graph.root_id,
            "document_metadata": doc_metadata,
            "document_graph": document_graph,
            "clean_filename": clean_filename
        }

    def run(self, filename: str) -> dict[str, Any]:
        """Process a single PDF and return its graph components.

        Returns a dict with keys: text_nodes, image_nodes, table_nodes,
        edges (hierarchy tree, list of (parent_id, child_id, weight) tuples),
        reference_edges (mentioning text -> media item, same tuple shape,
        not part of the tree),
        root_id (snippet_id of the document root: the title node, or the
        synthetic root if no title node was found),
        document_metadata (a Document),
        document_graph (the same graph as a DocumentGraph: one flat snippet
        list with globally unique ids, saved to <data_path>/graphs/ and
        reloadable with graph_store.load_graph),
        clean_filename.
        """
        return self._run(filename)


    def _get_text_value_for_snippet_type(self, node, snippet_type:str) -> str:
        if snippet_type == SnippetType.TEXT:
            return node.text
        elif snippet_type == SnippetType.IMAGE:
            return node.caption_text
        elif snippet_type == SnippetType.TABLE:
            return node.markdown_serialization
        else:
            raise ValueError(f"Unknown snippet type: {snippet_type}")

    def _node_to_snippet(
        self, node, snippet_type: str
    ) -> Snippet:
        """Convert a node (text, image, or table) to a Document Snippet.

        Args:
            node: TextSnippetNode or ImageSnippetNode or TableSnippetNode
            snippet_type: SnippetType.TEXT or SnippetType.IMAGE or SnippetType.TABLE
        """
        text_value = self._get_text_value_for_snippet_type(node, snippet_type)
        snippet = Snippet(
            snippet_id=node.snippet_id,
            type=snippet_type,
            document_id=str(node.document_id),
            sequence_no=node.sequence_no,
            label=node.label,
            level=node.level,
            level_label=node.level_label,
            parent_id=node.parent_id,
            is_grouped=node.is_grouped,
            region=getattr(node, "region", "body"),
            page_no=node.page_no,
            bbox=node.bbox,
            charspan=getattr(node, "charspan", None),
            provenance=node.provenance,
            text=text_value,
            docling_parent_ref=node.docling_parent_ref,
            docling_self_ref=node.docling_self_ref
        )
        return snippet

    def generate_snippets(self, return_snippets: bool = False) -> List[Snippet] | None:
        """
        Main method to run the entire extraction and graph construction pipeline for all documents.
        """
        all_snippets = []
        for file in tqdm(os.listdir(self.pdf_path)):
            result = self._run(file)

            # Convert nodes to snippets
            text_snippets = [
                self._node_to_snippet(tn, SnippetType.TEXT)
                for tn in result["text_nodes"]
            ]
            image_snippets = [
                self._node_to_snippet(inode, SnippetType.IMAGE)
                for inode in result["image_nodes"]
            ]
            table_snippets = [
                self._node_to_snippet(tn, SnippetType.TABLE)
                for tn in result["table_nodes"]
            ]
            snippets = text_snippets + image_snippets + table_snippets

            # default config is to save the snippets as json, but optionally can skip this step and just return the snippets
            if self.save_json:
                snippets_json_dir = os.path.join(self.data_path, "snippets/")
                os.makedirs(snippets_json_dir, exist_ok=True)
                with open(os.path.join(snippets_json_dir, f"{result['clean_filename']}_snippets.json"), "w") as f:
                    json.dump([snippet.model_dump() for snippet in snippets], f, indent=4)
            if return_snippets:
                all_snippets.extend(snippets)
        return all_snippets if return_snippets else None