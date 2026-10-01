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
from .document2graph_extractor import (
    DocumentGraphExtractor,
    build_snippet_graph,
    graph_from_docling,
)
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
    ExpansionPolicy,
    MergedEdge,
    MergedUnit,
    MergePolicy,
    MetadataExtractionConfig,
    MetadataFieldConfig,
    PageGeometry,
    PipelineFlags,
    Provenance,
    Snippet,
)
from .retrieval import (
    Calibration,
    expand,
    calibrate_min_weight,
    mean_unit_tokens,
    merge_units,
)
from .utils.ids import document_id_for

__all__ = [
    "BaselineExtractor",
    "Calibration",
    "DocumentGraphExtractor",
    "Chunk",
    "ChunkerConfig",
    "Document",
    "DocumentGraph",
    "DocumentMetadata",
    "EdgeWeightConfig",
    "ExpansionPolicy",
    "ExtractorConfig",
    "GraphSnippet",
    "MergePolicy",
    "MergedEdge",
    "MergedUnit",
    "MetadataExtractionConfig",
    "MetadataFieldConfig",
    "PageGeometry",
    "PipelineFlags",
    "Provenance",
    "Snippet",
    "build_snippet_graph",
    "calibrate_min_weight",
    "document_id_for",
    "expand",
    "graph_from_docling",
    "ensure_neo4j_constraints",
    "global_id",
    "load_graph",
    "mean_unit_tokens",
    "merge_units",
    "save_graph",
    "write_graph_to_neo4j",
]
