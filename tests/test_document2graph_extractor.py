import hashlib
import os

import networkx as nx
import pytest
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from document2graph.document2graph_extractor import DocumentGraphExtractor
from document2graph.models import ExtractorConfig, MetadataFieldConfig, MetadataExtractionConfig
from pydantic import BaseModel
from datetime import datetime, timezone
from neo4j import GraphDatabase



PDF_DIR = "tests/.test-pdf/"
DATA_PATH = "tests/.test-data/"


@pytest.fixture
def config() -> ExtractorConfig:
    # force CPU: docling's layout model needs float64, which MPS does not support
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
    )
    metadata_config = MetadataExtractionConfig(
        title_page=1,
        version=None,
        authors=MetadataFieldConfig(label="autoren", pages=(1, 1)),
        institutions=MetadataFieldConfig(label="institute", pages=(1, 1)),
        bibliography=MetadataFieldConfig(label="bibliografie", pages=(1, 1)),
        correspondence=MetadataFieldConfig(label="korrespondenzadresse", pages=(1, 1)),
    )
    return ExtractorConfig(
        pdf_path=PDF_DIR,
        data_path=DATA_PATH,
        document_type="Praxisempfehlung",
        save_json=True,
        pdfPipelineOptions=pipeline_options,
        metadata_config=metadata_config,
    )
class Snippet(BaseModel):
    """A node of the document graph layer, aligned with document2graph output.

    snippet_id is globally unique (hash of document_id + local_ref);
    local_ref is the document-local docling reference, e.g. "#/texts/12".
    """

    snippet_id: str
    local_ref: str
    document_id: str
    sequence_no: int
    label: str
    level: int
    level_label: str
    page_no: int
    bbox: dict | None = None
    text: str
    snippet_type: str = "text"
    line_heights: list[float] = []
    font_key: str | None = None
    level_height: float | None = None
    extracted_at: str


class DocumentGraph(BaseModel):
    """The document graph layer of a single document."""

    document_id: str
    filename: str
    title: str
    document_type: str = ""
    version: str = ""
    authors: str = ""
    institutions: str = ""
    root_id: str  # globally unique snippet_id of the document root (title node or synthetic root)
    snippets: list[Snippet]
    edges: list[tuple[str, str, float]]  # hierarchy tree: (parent snippet_id, child snippet_id, weight)
    reference_edges: list[tuple[str, str, float]] = []  # mentions: (text snippet_id, media snippet_id, weight)

def _to_document_graph(
        result: dict, document_id: str, filename: str, document_type: str
    ) -> DocumentGraph:
        meta = result["document_metadata"]
        extracted_at = datetime.now(timezone.utc).isoformat()

        snippets: list[Snippet] = []
        for node, snippet_type, text_attr in (
            *[(n, "text", "text") for n in result["text_nodes"]],
            *[(n, "image", "caption_text") for n in result["image_nodes"]],
            *[(n, "table", "markdown_serialization") for n in result["table_nodes"]],
        ):
            text = (getattr(node, text_attr, "") or "").replace("\x00", "")
            snippets.append(
                Snippet(
                    snippet_id=_global_id(document_id, node.snippet_id),
                    local_ref=node.snippet_id,
                    document_id=document_id,
                    sequence_no=node.sequence_no,
                    label=node.label,
                    level=node.level,
                    level_label=node.level_label,
                    page_no=node.page_no,
                    bbox=node.bbox.model_dump(mode="json") if node.bbox is not None else None,
                    text=text,
                    snippet_type=snippet_type,
                    line_heights=getattr(node, "line_heights", []),
                    font_key=getattr(node, "font_key", None),
                    level_height=getattr(node, "level_height", None),
                    extracted_at=extracted_at,
                )
            )

        # the root is usually the document title node and already part of the
        # snippets; only the synthetic fallback root needs to be materialized.
        root_local_ref = result["root_id"]
        root_id = _global_id(document_id, root_local_ref)
        if root_local_ref not in {s.local_ref for s in snippets}:
            snippets.append(
                Snippet(
                    snippet_id=root_id,
                    local_ref=root_local_ref,
                    document_id=document_id,
                    sequence_no=-1,
                    label="document_root",
                    level=-1,
                    level_label="Root",
                    page_no=0,
                    bbox=None,
                    text=meta.title or filename,
                    snippet_type="root",
                    extracted_at=extracted_at,
                )
            )

        edges = [
            (_global_id(document_id, parent), _global_id(document_id, child), float(weight))
            for parent, child, weight in result["edges"]
        ]
        reference_edges = [
            (_global_id(document_id, source), _global_id(document_id, target), float(weight))
            for source, target, weight in result["reference_edges"]
        ]
        return DocumentGraph(
            document_id=document_id,
            filename=filename,
            title=meta.title or filename,
            document_type=document_type,
            version=meta.metadata.version,
            authors=meta.metadata.authors,
            institutions=meta.metadata.institutions,
            root_id=root_id,
            snippets=snippets,
            edges=edges,
            reference_edges=reference_edges,
        )

def _global_id(document_id: str, local_ref: str) -> str:
    """Snippet ids from document2graph are document-local docling refs
    (e.g. "#/texts/12"); hash them with the document id to get a globally
    unique snippet_id."""
    return hashlib.sha256(f"{document_id}:{local_ref}".encode()).hexdigest()

def _write_graph_to_neo4j(driver, doc_graph: DocumentGraph) -> None:
        snippet_rows = [
            {
                "snippet_id": s.snippet_id,
                "local_ref": s.local_ref,
                "document_id": s.document_id,
                "sequence_no": s.sequence_no,
                "label": s.label,
                "level": s.level,
                "level_label": s.level_label,
                "page_no": s.page_no,
                "snippet_type": s.snippet_type,
                "line_heights": s.line_heights,
                "font_key": s.font_key or "",
                "level_height": s.level_height if s.level_height is not None else -1.0,
                "text_preview": s.text[:100] if s.text else "",
            }
            for s in doc_graph.snippets
        ]
        edge_rows = [
            {"parent": parent, "child": child, "weight": weight}
            for parent, child, weight in doc_graph.edges
        ]
        with driver.session() as session:
            session.run(
                "MERGE (d:Document {document_id: $document_id}) "
                "SET d.filename = $filename, d.title = $title, "
                "    d.document_type = $document_type, d.version = $version, "
                "    d.authors = $authors, d.institutions = $institutions",
                document_id=doc_graph.document_id,
                filename=doc_graph.filename,
                title=doc_graph.title,
                document_type=doc_graph.document_type,
                version=doc_graph.version,
                authors=doc_graph.authors,
                institutions=doc_graph.institutions,
            )
            session.run(
                "UNWIND $rows AS row "
                "MERGE (s:Snippet {snippet_id: row.snippet_id}) "
                "SET s += row",
                rows=snippet_rows,
            )
            session.run(
                "MATCH (d:Document {document_id: $document_id}) "
                "MATCH (r:Snippet {snippet_id: $root_id}) "
                "MERGE (d)-[:HAS_ROOT]->(r)",
                document_id=doc_graph.document_id,
                root_id=doc_graph.root_id,
            )
            session.run(
                "UNWIND $rows AS row "
                "MATCH (p:Snippet {snippet_id: row.parent}) "
                "MATCH (c:Snippet {snippet_id: row.child}) "
                "MERGE (p)-[e:HAS_CHILD]->(c) "
                "SET e.weight = row.weight",
                rows=edge_rows,
            )
            reference_rows = [
                {"source": source, "target": target, "weight": weight}
                for source, target, weight in doc_graph.reference_edges
            ]
            session.run(
                "UNWIND $rows AS row "
                "MATCH (t:Snippet {snippet_id: row.source}) "
                "MATCH (m:Snippet {snippet_id: row.target}) "
                "MERGE (t)-[e:REFERENCES]->(m) "
                "SET e.weight = row.weight",
                rows=reference_rows,
            )


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(PDF_DIR), reason="test PDFs not available")
def test_document2graph_extractor(config: ExtractorConfig):
    extractor = DocumentGraphExtractor(config)
    snippets = extractor.generate_snippets(return_snippets=True)

    # Assert that snippets are generated
    assert snippets is not None
    assert len(snippets) > 0
    # Assert that each snippet has the required fields
    for snippet in snippets:
        assert hasattr(snippet, "snippet_id")
        assert hasattr(snippet, "type")
        assert hasattr(snippet, "document_id")
        assert hasattr(snippet, "sequence_no")
        assert hasattr(snippet, "label")
        assert hasattr(snippet, "level")
        assert hasattr(snippet, "level_label")
        assert hasattr(snippet, "parent_id")
        assert hasattr(snippet, "is_grouped")
        assert hasattr(snippet, "page_no")
        assert hasattr(snippet, "bbox")
        assert hasattr(snippet, "text")
        assert hasattr(snippet, "docling_parent_ref")
        assert hasattr(snippet, "docling_self_ref")

@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(PDF_DIR), reason="test PDFs not available")
def test_document2graph_extractor_and_neo4j(config: ExtractorConfig):
    extractor = DocumentGraphExtractor(config)

    with GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password"))
    ) as driver:
        for file in os.listdir(config.pdf_path):
            if file.endswith(".pdf"):
                result = extractor.run(file)
                assert result is not None
                assert "text_nodes" in result
                assert "image_nodes" in result
                assert "table_nodes" in result
                assert "edges" in result
                assert "reference_edges" in result
                assert "root_id" in result
                assert "document_metadata" in result
                assert "clean_filename" in result

                # Convert to DocumentGraph
                doc_graph = _to_document_graph(
                    result=result,
                    document_id=hashlib.sha256(file.encode()).hexdigest(),
                    filename=file,
                    document_type="Praxisempfehlung",
                )

                # Write to Neo4j
                _write_graph_to_neo4j(driver, doc_graph)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(PDF_DIR), reason="test PDFs not available")
def test_document2graph_is_connected_and_weighted(config: ExtractorConfig):
    extractor = DocumentGraphExtractor(config)
    extractor.generate_snippets()

    graph_dir = os.path.join(config.data_path, "nx_graphs/")
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith(".gexf")]
    assert len(graph_files) > 0
    for graph_file in graph_files:
        graph = nx.read_gexf(os.path.join(graph_dir, graph_file))
        assert nx.is_weakly_connected(graph), f"{graph_file} has disconnected components"
        assert all("weight" in data for _, _, data in graph.edges(data=True)), f"{graph_file} has unweighted edges"


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_document2graph_extractor.py"])
