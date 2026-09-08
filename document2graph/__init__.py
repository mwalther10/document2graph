"""Extract hierarchical document graphs and baseline chunks from PDF files.

Two complementary representations of the same document:

- ``DocumentGraphExtractor`` builds a hierarchical graph of text, image and
  table nodes joined by weighted structural edges.
- ``BaselineExtractor`` produces flat, token-bounded chunks as a retrieval
  baseline to compare that graph against.

Both share one ``ExtractorConfig``, derive the same ``document_id`` for a given
PDF, and expose provenance the same way (``unit.provenance``), so nodes and
chunks can be aligned back to the page they came from.
"""

from .baseline_extractor import BaselineExtractor
from .document2graph_extractor import DocumentGraphExtractor
from .graph_store import (
    DocumentGraph,
    GraphSnippet,
    ensure_neo4j_constraints,
    global_id,
    load_graph,
    save_graph,
    write_graph_to_neo4j,
)
from .models import (
    Chunk,
    ChunkerConfig,
    Document,
    DocumentMetadata,
    EdgeWeightConfig,
    ExtractorConfig,
    MetadataExtractionConfig,
    MetadataFieldConfig,
    PipelineFlags,
    Provenance,
    RelevancyWeightConfig,
    Snippet,
)
from .utils.ids import document_id_for

__all__ = [
    "BaselineExtractor",
    "DocumentGraphExtractor",
    "Chunk",
    "ChunkerConfig",
    "Document",
    "DocumentGraph",
    "DocumentMetadata",
    "EdgeWeightConfig",
    "ExtractorConfig",
    "GraphSnippet",
    "MetadataExtractionConfig",
    "MetadataFieldConfig",
    "PipelineFlags",
    "Provenance",
    "RelevancyWeightConfig",
    "Snippet",
    "document_id_for",
    "ensure_neo4j_constraints",
    "global_id",
    "load_graph",
    "save_graph",
    "write_graph_to_neo4j",
]
