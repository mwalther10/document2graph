"""Build a document graph from an already-parsed document.

``DocumentGraphExtractor`` is the batch entry point: it owns a corpus directory,
decides where the intermediate files go, and writes them. That is the wrong shape
for a harness running many conditions over one corpus, which has already parsed
the document, wants the graph as a value rather than as a directory of files, and
varies only what happens *after* the parse.

These two functions are that seam. They take the parsed document and return the
graph, so an ablation over ``PipelineFlags`` or ``EdgeWeightConfig`` reuses one
parse and never touches the filesystem::

    from docling_core.types.doc.document import DoclingDocument
    from docling_parse.pdf_parser import DoclingPdfParser
    from document2graph import graph_from_docling

    docling_doc = DoclingDocument.load_from_json(cached_json)   # the expensive parse, cached
    pdf_doc = DoclingPdfParser().load(path_or_stream=pdf)       # page geometry and fonts

    graph = graph_from_docling(docling_doc, pdf_doc, "my_document",
                               flags=PipelineFlags(stitch_continuations=False))

``pdf_doc`` is still required, and is the reason a condition sweep needs the PDF
alongside the cached parse. Graph construction reads three things from it that
the ``DoclingDocument`` does not carry: the line and character cells behind line
heights and font keys, and the drawn shapes behind page blocks and regions. It is
cheap to open (the loader is lazy; pages are parsed on first use and cached
thereafter, around 0.1s each), but it does need the PDF's bytes.
"""

from docling_core.types.doc.document import DoclingDocument
from docling_parse.pdf_parser import PdfDocument

from ..graph_store import DocumentGraph
from ..models.Document import Document
from ..models.DocumentMetadata import MetadataExtractionConfig
from ..models.EdgeWeightConfig import EdgeWeightConfig
from ..models.PipelineFlags import PipelineFlags
from .snippet_graph_constructor import SnippetGraph, SnippetGraphConstructor


def build_snippet_graph(
    docling_doc: DoclingDocument,
    pdf_doc: PdfDocument,
    filename: str,
    document_type: str = "",
    metadata_config: MetadataExtractionConfig | None = None,
    edge_weights: EdgeWeightConfig | None = None,
    flags: PipelineFlags | None = None,
    save_gexf_to: str = "",
) -> tuple[SnippetGraph, Document]:
    """The raw constructor result and the document metadata it extracted.

    ``filename`` identifies the document: the extension is stripped before
    hashing, so the name and its stem derive the same ``document_id``.
    """
    constructor = SnippetGraphConstructor(
        pdf_doc,
        docling_doc,
        filename,
        document_type=document_type,
        metadata_config=metadata_config,
        edge_weights=edge_weights,
        flags=flags,
    )
    return constructor.get_graph(save_to=save_gexf_to), constructor.document_metadata


def graph_from_docling(
    docling_doc: DoclingDocument,
    pdf_doc: PdfDocument,
    filename: str,
    document_type: str = "",
    metadata_config: MetadataExtractionConfig | None = None,
    edge_weights: EdgeWeightConfig | None = None,
    flags: PipelineFlags | None = None,
    save_gexf_to: str = "",
) -> DocumentGraph:
    """The graph of one parsed document, with globally unique ids.

    The same object ``DocumentGraphExtractor.run()`` returns under
    ``"document_graph"`` and ``graph_store.load_graph`` reads back, built without
    a corpus directory: nothing is written unless ``save_gexf_to`` is set.
    """
    graph, document = build_snippet_graph(
        docling_doc,
        pdf_doc,
        filename,
        document_type=document_type,
        metadata_config=metadata_config,
        edge_weights=edge_weights,
        flags=flags,
        save_gexf_to=save_gexf_to,
    )
    return DocumentGraph.from_snippet_graph(graph, document)
