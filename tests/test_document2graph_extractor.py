import os

import networkx as nx
import pytest
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from document2graph.document2graph_extractor import DocumentGraphExtractor
from document2graph.models import ExtractorConfig

PDF_DIR = "tests/.test-pdf/"


@pytest.fixture
def config(tmp_path) -> ExtractorConfig:
    # force CPU: docling's layout model needs float64, which MPS does not support
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
    )
    return ExtractorConfig(
        pdf_path=PDF_DIR,
        data_path=str(tmp_path),
        document_type="Praxisempfehlung",
        save_json=True,
        pdfPipelineOptions=pipeline_options,
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
