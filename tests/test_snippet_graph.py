import networkx as nx
import pytest
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import RefItem

from document_graph.document_graph_extractor.snippet_graph_constructor import (
    ROOT_NODE_ID,
    SnippetGraphConstructor,
)
from document_graph.models import (
    Document,
    EdgeWeightConfig,
    ImageSnippetNode,
    TextSnippetNode,
)


def make_text_node(idx: int, level: int, level_label: str, parent_id: str | None, is_grouped: bool = False) -> TextSnippetNode:
    return TextSnippetNode(
        snippet_id=f"#/texts/{idx}",
        document_id="doc-1",
        label="text",
        sequence_no=idx,
        level=level,
        level_label=level_label,
        parent_id=parent_id,
        docling_parent_ref=RefItem(**{"$ref": "#/body"}),
        docling_self_ref=RefItem(**{"$ref": f"#/texts/{idx}"}),
        is_grouped=is_grouped,
        text=f"text {idx}",
        bbox=BoundingBox(l=0, t=1, r=1, b=0),
        charspan=(0, 6),
        page_no=1,
    )


def make_image_node(idx: int, level_label: str, parent_id: str) -> ImageSnippetNode:
    return ImageSnippetNode(
        snippet_id=f"#/pictures/{idx}",
        document_id="doc-1",
        label="picture",
        sequence_no=idx,
        level=2,
        level_label=level_label,
        parent_id=parent_id,
        docling_parent_ref=None,
        docling_self_ref=None,
        caption_text="a caption",
        bbox=BoundingBox(l=0, t=1, r=1, b=0),
        page_no=1,
    )


@pytest.fixture
def constructor() -> SnippetGraphConstructor:
    # bypass __init__: these methods only need edge_weights and document metadata
    c = SnippetGraphConstructor.__new__(SnippetGraphConstructor)
    c.edge_weights = EdgeWeightConfig()
    c._document_metadata = Document(document_id="doc-1", title="Test Doc")
    return c


def test_edge_weights_by_category(constructor: SnippetGraphConstructor):
    weights = constructor.edge_weights
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    subheading = make_text_node(1, level=1, level_label="Heading", parent_id=heading.snippet_id)
    body = make_text_node(2, level=2, level_label="Body", parent_id=subheading.snippet_id)
    bullet = make_text_node(3, level=2, level_label="Body", parent_id=body.snippet_id, is_grouped=True)
    image = make_image_node(0, level_label="Body", parent_id=body.snippet_id)
    orphan_image = make_image_node(1, level_label="Unreferenced Image", parent_id=heading.snippet_id)

    assert constructor.compute_edge_weight(heading, subheading) == weights.section
    assert constructor.compute_edge_weight(subheading, body) == weights.text
    assert constructor.compute_edge_weight(body, bullet) == weights.list_item
    assert constructor.compute_edge_weight(body, image) == weights.media
    assert constructor.compute_edge_weight(heading, orphan_image) == weights.unreferenced_media


def test_custom_edge_weights_are_used(constructor: SnippetGraphConstructor):
    constructor.edge_weights = EdgeWeightConfig(section=5.0, text=2.5)
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    subheading = make_text_node(1, level=1, level_label="Heading", parent_id=heading.snippet_id)
    body = make_text_node(2, level=2, level_label="Body", parent_id=subheading.snippet_id)

    assert constructor.compute_edge_weight(heading, subheading) == 5.0
    assert constructor.compute_edge_weight(subheading, body) == 2.5


def test_orphans_and_dangling_parents_attach_to_root(constructor: SnippetGraphConstructor):
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    body = make_text_node(1, level=2, level_label="Body", parent_id=heading.snippet_id)
    dangling = make_text_node(2, level=2, level_label="Body", parent_id="#/does/not/exist")
    nodes = [heading, body, dangling]

    edges = constructor.construct_snippet_edges(nodes)

    assert (ROOT_NODE_ID, heading.snippet_id, constructor.edge_weights.root) in edges
    assert (heading.snippet_id, body.snippet_id, constructor.edge_weights.text) in edges
    assert (ROOT_NODE_ID, dangling.snippet_id, constructor.edge_weights.root) in edges
    # every node has exactly one incoming edge
    assert len(edges) == len(nodes)


def test_graph_is_connected(constructor: SnippetGraphConstructor):
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    body = make_text_node(1, level=2, level_label="Body", parent_id=heading.snippet_id)
    # cycle detached from the root: 2 -> 3 -> 2
    cycle_a = make_text_node(2, level=1, level_label="Heading", parent_id="#/texts/3")
    cycle_b = make_text_node(3, level=2, level_label="Body", parent_id="#/texts/2")
    nodes = [heading, body, cycle_a, cycle_b]

    edges = constructor.construct_snippet_edges(nodes)
    edges += constructor.connect_components(nodes, edges)
    graph = constructor.build_nx_graph([heading, body, cycle_a, cycle_b], [], [], edges)

    assert nx.is_weakly_connected(graph)
    # the cycle is attached via its structurally highest node
    assert graph.has_edge(ROOT_NODE_ID, cycle_a.snippet_id)
    assert all("weight" in data for _, _, data in graph.edges(data=True))
