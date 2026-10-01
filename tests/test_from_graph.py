"""Rescoring a saved graph's edges without the PDF."""

import pytest

from document2graph.edge_weights import edge_context_from_graph, recompute_weights
from document2graph.edge_weights.context import KIND_IMAGE, KIND_TABLE, KIND_TEXT
from document2graph.graph_store import DocumentGraph, GraphSnippet, edge_key
from document2graph.models import EdgeWeightConfig

DOC = "doc-1"


def snippet(ref, text, sequence_no, level_label="body", snippet_type="text",
            is_grouped=False) -> GraphSnippet:
    return GraphSnippet(
        snippet_id=ref, local_ref=ref, document_id=DOC, sequence_no=sequence_no,
        label="text", snippet_type=snippet_type, level=0, level_label=level_label,
        page_no=1, text=text, is_grouped=is_grouped,
        extracted_at="2026-01-01T00:00:00Z",
    )


@pytest.fixture
def graph() -> DocumentGraph:
    return DocumentGraph(
        document_id=DOC, filename="doc.pdf", title="t", root_id="#/h",
        snippets=[
            snippet("#/h", "Therapie", 0, level_label="Heading"),
            snippet("#/p", "Die Dosis wird angepasst.", 1),
            snippet("#/b", "zweimal", 2, is_grouped=True),
            snippet("#/tab", "| a | b |", 3, snippet_type="table"),
            snippet("#/fig", "Abbildung 1", 4, snippet_type="image"),
        ],
        edges=[("#/h", "#/p", 0.8), ("#/p", "#/b", 0.6), ("#/h", "#/tab", 0.3)],
        reference_edges=[("#/p", "#/fig", 0.7)],
        edge_relations={
            edge_key("#/h", "#/p"): "text",
            edge_key("#/p", "#/b"): "list_item",
            edge_key("#/h", "#/tab"): "unreferenced_media",
            edge_key("#/p", "#/fig"): "reference",
        },
    )


def test_the_context_carries_what_the_metrics_read(graph):
    ctx = edge_context_from_graph(graph)
    assert ctx.unit("#/h").is_heading
    assert ctx.unit("#/b").is_grouped
    assert ctx.unit("#/tab").kind == KIND_TABLE
    assert ctx.unit("#/fig").kind == KIND_IMAGE
    assert ctx.unit("#/p").kind == KIND_TEXT
    assert ctx.text("#/p") == "Die Dosis wird angepasst."


def test_reference_edges_are_scored_alongside_the_tree(graph):
    """Construction weights both kinds in one pass so a metric that normalizes
    against the strongest edge normalizes against the same set."""
    ctx = edge_context_from_graph(graph)
    assert ("#/p", "#/fig") in ctx.edges
    assert len(ctx.edges) == 4
    assert len(edge_context_from_graph(graph, include_reference_edges=False).edges) == 3


def test_the_stored_relation_is_used_not_one_guessed_from_the_nodes(graph):
    """An unreferenced_media edge carries the highest supply prior (1.00); read off
    the nodes alone it would look like an ordinary media_figure edge (0.50)."""
    ctx = edge_context_from_graph(graph)
    assert ctx.relation(("#/h", "#/tab")) == "unreferenced_media"
    assert ctx.relation(("#/p", "#/fig")) == "reference"


def test_rescoring_rewrites_both_edge_kinds(graph):
    rescored = recompute_weights(graph, EdgeWeightConfig(metric="uniform"))
    assert [weight for _, _, weight in rescored.edges] == [1.0, 1.0, 1.0]
    assert [weight for _, _, weight in rescored.reference_edges] == [1.0]
    # the original is untouched
    assert [weight for _, _, weight in graph.edges] == [0.8, 0.6, 0.3]


def test_rescoring_reproduces_the_structural_weights_it_was_built_with(graph):
    rescored = recompute_weights(graph, EdgeWeightConfig(metric="structural"))
    assert dict(((p, c), w) for p, c, w in rescored.edges) == {
        ("#/h", "#/p"): 0.8, ("#/p", "#/b"): 0.6, ("#/h", "#/tab"): 0.3,
    }


def test_rescoring_a_graph_with_no_relation_table_is_refused(graph):
    """Guessing the relations would turn an unreferenced_media edge into a media
    one, which is wrong in a way that is invisible in the output."""
    stale = graph.model_copy(update={"edge_relations": {}, "schema_version": 2})
    with pytest.raises(ValueError, match="edge relations"):
        recompute_weights(stale, EdgeWeightConfig(metric="structural"))
