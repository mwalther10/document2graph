import os

import networkx as nx
import pytest
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from document2graph.document2graph_extractor import DocumentGraphExtractor
from document2graph.graph_store import ensure_neo4j_constraints, load_graph, write_graph_to_neo4j
from document2graph.models import ExtractorConfig, MetadataFieldConfig, MetadataExtractionConfig
from document2graph.utils.ids import document_id_for

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


def _neo4j_driver():
    """A driver for the configured Neo4j, or None when it is not reachable."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password"))
    )
    try:
        driver.verify_connectivity()
    except Exception:
        driver.close()
        return None
    return driver


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
        # every unit exposes its PDF locations the same way
        assert snippet.provenance
        assert all(p.page_no > 0 for p in snippet.provenance)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(PDF_DIR), reason="test PDFs not available")
def test_document_graph_is_saved_and_reloadable(config: ExtractorConfig):
    """The json graph round-trips, so a harness reads a corpus back without re-parsing."""
    extractor = DocumentGraphExtractor(config)
    pdf = next(f for f in sorted(os.listdir(config.pdf_path)) if f.endswith(".pdf"))
    result = extractor.run(pdf)

    document_graph = result["document_graph"]
    assert document_graph.document_id == document_id_for(pdf)
    assert document_graph.snippets

    clean_filename = result["clean_filename"]
    reloaded = load_graph(os.path.join(config.data_path, "graphs", f"{clean_filename}_graph.json"))
    assert reloaded == document_graph
    # the fields GEXF drops survive the round-trip
    assert any(s.bbox is not None and s.provenance for s in reloaded.snippets)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(PDF_DIR), reason="test PDFs not available")
def test_document2graph_extractor_and_neo4j(config: ExtractorConfig):
    driver = _neo4j_driver()
    if driver is None:
        pytest.skip("no reachable Neo4j at NEO4J_URI")

    extractor = DocumentGraphExtractor(config)
    with driver:
        ensure_neo4j_constraints(driver)
        for file in os.listdir(config.pdf_path):
            if file.endswith(".pdf"):
                result = extractor.run(file)
                assert result is not None
                for key in ("text_nodes", "image_nodes", "table_nodes", "edges",
                            "reference_edges", "root_id", "document_metadata",
                            "document_graph", "clean_filename"):
                    assert key in result

                write_graph_to_neo4j(driver, result["document_graph"])


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
