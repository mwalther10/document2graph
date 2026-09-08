import pytest
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import RefItem

from document2graph.document2graph_extractor.snippet_graph_constructor import (
    ROOT_NODE_ID,
    SnippetGraph,
)
from document2graph.graph_store import DocumentGraph, global_id, load_graph, save_graph
from document2graph.models import (
    Document,
    DocumentMetadata,
    ImageSnippetNode,
    Provenance,
    TextSnippetNode,
)
from document2graph.utils.ids import chunk_id_for, document_id_for

DOC_ID = document_id_for("report.pdf")


def make_document(title: str = "A title") -> Document:
    return Document(
        document_id=DOC_ID,
        document_type="report",
        filename="report.pdf",
        title=title,
        metadata=DocumentMetadata(version="2025", authors="#/texts/3", institutions="#/texts/4"),
    )


def make_text_node(idx: int, level: int, parent_id: str | None, page_no: int = 1) -> TextSnippetNode:
    return TextSnippetNode(
        snippet_id=f"#/texts/{idx}",
        document_id=DOC_ID,
        label="text",
        sequence_no=idx,
        level=level,
        level_label="Heading" if level == 0 else "body",
        parent_id=parent_id,
        docling_parent_ref=RefItem(**{"$ref": "#/body"}),
        docling_self_ref=RefItem(**{"$ref": f"#/texts/{idx}"}),
        text=f"text {idx}",
        bbox=BoundingBox(l=0, t=10, r=100, b=0, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 6),
        page_no=page_no,
        line_heights=[9.5],
        font_key="F1",
        # a paragraph stitched across a column break: two locations, one node
        provenance=[
            Provenance(page_no=page_no, bbox=BoundingBox(l=0, t=10, r=100, b=0), charspan=(0, 3)),
            Provenance(page_no=page_no + 1, bbox=BoundingBox(l=0, t=800, r=100, b=700), charspan=(3, 6)),
        ],
    )


def make_image_node(idx: int, parent_id: str | None) -> ImageSnippetNode:
    return ImageSnippetNode(
        snippet_id=f"#/pictures/{idx}",
        document_id=DOC_ID,
        label="picture",
        sequence_no=idx,
        level=2,
        level_label="figure",
        parent_id=parent_id,
        docling_parent_ref=RefItem(**{"$ref": "#/body"}),
        docling_self_ref=RefItem(**{"$ref": f"#/pictures/{idx}"}),
        caption_text=f"Abbildung {idx}",
        bbox=BoundingBox(l=0, t=200, r=300, b=100, coord_origin=CoordOrigin.BOTTOMLEFT),
        page_no=2,
        provenance=[Provenance(page_no=2, bbox=BoundingBox(l=0, t=200, r=300, b=100))],
    )


@pytest.fixture
def snippet_graph() -> SnippetGraph:
    title = make_text_node(0, level=0, parent_id=None)
    body = make_text_node(1, level=1, parent_id="#/texts/0")
    image = make_image_node(0, parent_id="#/texts/0")
    return SnippetGraph(
        text_nodes=[title, body],
        image_nodes=[image],
        table_nodes=[],
        edges=[("#/texts/0", "#/texts/1", 0.8), ("#/texts/0", "#/pictures/0", 0.9)],
        reference_edges=[("#/texts/1", "#/pictures/0", 0.7)],
        root_id="#/texts/0",
    )


def test_document_id_is_stable_and_independent_of_path_and_extension():
    assert document_id_for("report.pdf") == document_id_for("report.pdf")
    assert document_id_for("data/pdfs/report.pdf") == document_id_for("report.pdf")
    assert document_id_for("report.pdf") == document_id_for("report")
    assert document_id_for("other.pdf") != document_id_for("report.pdf")


def test_chunk_ids_are_deterministic_and_variant_specific():
    assert chunk_id_for(DOC_ID, 3) == chunk_id_for(DOC_ID, 3)
    assert chunk_id_for(DOC_ID, 3) != chunk_id_for(DOC_ID, 4)
    assert chunk_id_for(DOC_ID, 3) != chunk_id_for(DOC_ID, 3, variant="enriched")


def test_global_id_is_stable_and_document_scoped():
    assert global_id(DOC_ID, "#/texts/12") == global_id(DOC_ID, "#/texts/12")
    assert global_id(DOC_ID, "#/texts/12") != global_id(document_id_for("other.pdf"), "#/texts/12")


def test_from_snippet_graph_carries_every_node_and_edge(snippet_graph: SnippetGraph):
    graph = DocumentGraph.from_snippet_graph(snippet_graph, make_document())

    assert len(graph.snippets) == 3
    assert {s.snippet_type for s in graph.snippets} == {"text", "image"}
    assert graph.root_id == global_id(DOC_ID, "#/texts/0")
    assert graph.title == "A title"
    assert graph.version == "2025"
    # ids are global, and a child's parent_id resolves to its parent's snippet_id
    body = next(s for s in graph.snippets if s.local_ref == "#/texts/1")
    assert body.snippet_id == global_id(DOC_ID, "#/texts/1")
    assert body.parent_id == graph.root_id
    assert graph.edges == [
        (global_id(DOC_ID, "#/texts/0"), global_id(DOC_ID, "#/texts/1"), 0.8),
        (global_id(DOC_ID, "#/texts/0"), global_id(DOC_ID, "#/pictures/0"), 0.9),
    ]
    assert graph.reference_edges == [
        (global_id(DOC_ID, "#/texts/1"), global_id(DOC_ID, "#/pictures/0"), 0.7)
    ]


def test_synthetic_root_is_materialized_when_no_title_node(snippet_graph: SnippetGraph):
    rootless = snippet_graph._replace(root_id=ROOT_NODE_ID)
    graph = DocumentGraph.from_snippet_graph(rootless, make_document())

    assert len(graph.snippets) == 4
    root = next(s for s in graph.snippets if s.snippet_id == graph.root_id)
    assert root.local_ref == ROOT_NODE_ID
    assert root.snippet_type == "root"
    assert root.text == "A title"
    assert root.parent_id is None


def test_synthetic_root_falls_back_to_the_filename(snippet_graph: SnippetGraph):
    rootless = snippet_graph._replace(root_id=ROOT_NODE_ID)
    graph = DocumentGraph.from_snippet_graph(rootless, make_document(title=""))

    root = next(s for s in graph.snippets if s.snippet_id == graph.root_id)
    assert root.text == "report.pdf"


def test_provenance_survives_and_keeps_every_location(snippet_graph: SnippetGraph):
    graph = DocumentGraph.from_snippet_graph(snippet_graph, make_document())

    body = next(s for s in graph.snippets if s.local_ref == "#/texts/1")
    # the stitched paragraph reports both halves, not just prov[0]
    assert [p.page_no for p in body.provenance] == [1, 2]
    assert body.charspan == (0, 6)
    image = next(s for s in graph.snippets if s.local_ref == "#/pictures/0")
    assert [p.page_no for p in image.provenance] == [2]
    assert image.charspan is None


def test_save_and_load_graph_round_trip(tmp_path, snippet_graph: SnippetGraph):
    graph = DocumentGraph.from_snippet_graph(snippet_graph, make_document())
    path = save_graph(graph, str(tmp_path / "nested" / "report_graph.json"))

    reloaded = load_graph(path)
    assert reloaded == graph
    assert reloaded.model_dump() == graph.model_dump()
