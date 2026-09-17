"""Growing a retrieved set using the graph."""

import pytest

from document2graph.graph_store import DocumentGraph, GraphSnippet, edge_key
from document2graph.models import ExpansionPolicy, Provenance
from document2graph.retrieval import expand

DOC = "doc-1"


def snippet(ref, text, sequence_no, level_label="body", snippet_type="text") -> GraphSnippet:
    return GraphSnippet(
        snippet_id=ref, local_ref=ref, document_id=DOC, sequence_no=sequence_no,
        label="text", snippet_type=snippet_type, level=0, level_label=level_label,
        page_no=1, text=text, provenance=[Provenance(page_no=1)],
        extracted_at="2026-01-01T00:00:00Z",
    )


@pytest.fixture
def graph() -> DocumentGraph:
    """A heading over three bullets and a paragraph that cites a table.

        H   "Therapieziele"
         +- b1  "HbA1c unter 7%"        H->b1  0.8
         +- b2  "Blutdruck unter 140"   H->b2  0.8
         +- b3  "LDL unter 100"         H->b3  0.8
         +- p   "Siehe Tab. 1."         H->p   0.8   p --references--> tab
        T   "Nebenwirkungen"  (a second section, so not everything shares one parent)
         +- q   "Unterzuckerung"        T->q   0.8
    """
    return DocumentGraph(
        document_id=DOC, filename="doc.pdf", title="t", root_id="#/h",
        snippets=[
            snippet("#/h", "Therapieziele", 0, level_label="Heading"),
            snippet("#/b1", "HbA1c unter 7%", 1),
            snippet("#/b2", "Blutdruck unter 140", 2),
            snippet("#/b3", "LDL unter 100", 3),
            snippet("#/p", "Siehe Tab. 1.", 4),
            snippet("#/t", "Nebenwirkungen", 5, level_label="Heading"),
            snippet("#/q", "Unterzuckerung", 6),
            snippet("#/tab", "| Ziel | Wert |", 7, snippet_type="table"),
        ],
        edges=[("#/h", "#/b1", 0.8), ("#/h", "#/b2", 0.8), ("#/h", "#/b3", 0.8),
               ("#/h", "#/p", 0.8), ("#/t", "#/q", 0.8), ("#/h", "#/t", 1.0)],
        reference_edges=[("#/p", "#/tab", 0.7)],
        edge_relations={
            edge_key("#/h", "#/b1"): "list_item", edge_key("#/h", "#/b2"): "list_item",
            edge_key("#/h", "#/b3"): "list_item", edge_key("#/h", "#/p"): "text",
            edge_key("#/t", "#/q"): "text", edge_key("#/h", "#/t"): "section",
            edge_key("#/p", "#/tab"): "reference",
        },
    )


def ids(results):
    return [result.hit_id for result in results]


# --- the no-op ---------------------------------------------------------------

def test_none_returns_the_hits_unchanged(graph):
    hits = [("#/b1", 0.9), ("#/q", 0.5)]
    results = expand(graph, hits, ExpansionPolicy(mode="none"))
    assert ids(results) == ["#/b1", "#/q"]
    assert all(result.unit.member_ids == [result.hit_id] for result in results)
    assert all(result.added_tokens == 0 for result in results)


def test_results_come_back_in_score_order(graph):
    hits = [("#/q", 0.2), ("#/b1", 0.9), ("#/b2", 0.5)]
    assert ids(expand(graph, hits, ExpansionPolicy(mode="none"))) == ["#/b1", "#/b2", "#/q"]


def test_a_hit_the_graph_does_not_contain_is_dropped(graph):
    results = expand(graph, [("#/b1", 0.9), ("#/gone", 0.8)], ExpansionPolicy(mode="none"))
    assert ids(results) == ["#/b1"]


# --- simple expansions -------------------------------------------------------

def test_parent_adds_the_parent_ahead_of_the_hit(graph):
    result = expand(graph, [("#/b1", 0.9)], ExpansionPolicy(mode="parent"))[0]
    assert result.unit.member_ids == ["#/h", "#/b1"]
    assert result.unit.text.startswith("Therapieziele")
    assert result.reason == "parent"
    assert result.added_ids == ["#/h"]
    assert result.added_tokens > 0


def test_parent_is_skipped_when_the_edge_is_too_weak(graph):
    policy = ExpansionPolicy(mode="parent", min_edge_weight=0.9)
    result = expand(graph, [("#/b1", 0.9)], policy)[0]
    assert result.unit.member_ids == ["#/b1"]
    assert result.reason == "hit"


def test_siblings_are_added_in_document_order(graph):
    result = expand(graph, [("#/b2", 0.9)], ExpansionPolicy(mode="siblings"))[0]
    # "#/t" is a sibling too: a subheading sits under the same parent as the
    # paragraphs beside it, which is the tree being read correctly, not an extra
    assert result.unit.member_ids == ["#/b1", "#/b2", "#/b3", "#/p", "#/t"]


def test_subtree_adds_what_hangs_below_a_retrieved_heading(graph):
    result = expand(graph, [("#/t", 0.9)], ExpansionPolicy(mode="subtree"))[0]
    assert result.unit.member_ids == ["#/t", "#/q"]


def test_reference_edges_follow_a_mention_to_its_table(graph):
    """The one expansion no flat representation can do: nothing but the graph
    connects a paragraph to the table it cites."""
    result = expand(graph, [("#/p", 0.9)], ExpansionPolicy(mode="reference_edges"))[0]
    assert "#/tab" in result.unit.member_ids
    assert result.reason == "reference_edges"


def test_reference_edges_are_followed_from_the_table_back_to_the_mention(graph):
    result = expand(graph, [("#/tab", 0.9)], ExpansionPolicy(mode="reference_edges"))[0]
    assert "#/p" in result.unit.member_ids


# --- auto-merge --------------------------------------------------------------

def test_enough_retrieved_siblings_are_returned_as_their_parent(graph):
    hits = [("#/b1", 0.9), ("#/b2", 0.8), ("#/q", 0.4)]
    results = expand(graph, hits, ExpansionPolicy(mode="auto_merge", min_siblings=2))

    merged = results[0]
    assert merged.hit_id == "#/h"
    assert merged.reason == "auto_merge"
    assert merged.unit.member_ids == ["#/h", "#/b1", "#/b2"]
    # the parent takes the rank of its best child, not the sum of them
    assert merged.score == 0.9
    assert merged.absorbed_ids == ["#/b2"]
    # the unrelated hit is untouched, and b2 is not also returned on its own
    assert ids(results) == ["#/h", "#/q"]


def test_one_retrieved_child_is_not_enough(graph):
    hits = [("#/b1", 0.9), ("#/q", 0.4)]
    results = expand(graph, hits, ExpansionPolicy(mode="auto_merge", min_siblings=2))
    assert ids(results) == ["#/b1", "#/q"]
    assert all(result.reason == "hit" for result in results)


def test_min_siblings_is_configurable(graph):
    hits = [("#/b1", 0.9), ("#/b2", 0.8)]
    assert ids(expand(graph, hits, ExpansionPolicy(mode="auto_merge", min_siblings=3))) == \
        ["#/b1", "#/b2"]
    assert ids(expand(graph, hits, ExpansionPolicy(mode="auto_merge", min_siblings=2))) == ["#/h"]


def test_a_weak_parent_does_not_collapse_its_children(graph):
    hits = [("#/b1", 0.9), ("#/b2", 0.8)]
    policy = ExpansionPolicy(mode="auto_merge", min_siblings=2, min_edge_weight=0.9)
    assert ids(expand(graph, hits, policy)) == ["#/b1", "#/b2"]


def test_auto_merge_does_not_walk_on_up_to_the_root(graph):
    """Collapsing repeatedly would carry a section to the document root as soon as
    two of its paragraphs matched, which says something about the tree rather than
    about the query."""
    hits = [("#/b1", 0.9), ("#/b2", 0.8), ("#/b3", 0.7), ("#/q", 0.6)]
    results = expand(graph, hits, ExpansionPolicy(mode="auto_merge", min_siblings=2))
    # b1..b3 collapse into #/h; #/q stays, and #/h does not then swallow #/t
    assert ids(results) == ["#/h", "#/q"]


def test_a_parent_that_was_itself_retrieved_is_not_duplicated(graph):
    hits = [("#/h", 0.9), ("#/b1", 0.8), ("#/b2", 0.7)]
    results = expand(graph, hits, ExpansionPolicy(mode="auto_merge", min_siblings=2))
    assert ids(results).count("#/h") == 1


def test_dedupe_off_returns_the_children_alongside_the_parent(graph):
    hits = [("#/b1", 0.9), ("#/b2", 0.8)]
    results = expand(graph, hits, ExpansionPolicy(mode="auto_merge", dedupe=False))
    assert ids(results) == ["#/b1", "#/b2"]


# --- budgets -----------------------------------------------------------------

def test_the_expansion_budget_is_spent_on_the_best_ranked_hits(graph):
    """Hits are grown in rank order, so a tight allowance grows the top of the
    result and leaves the tail bare rather than spreading thinly over everything."""
    hits = [("#/b1", 0.9), ("#/b2", 0.8), ("#/b3", 0.7)]
    # adding the heading to any one bullet costs 4 tokens, so 5 buys exactly one
    policy = ExpansionPolicy(mode="parent", max_added_tokens=5)
    results = expand(graph, hits, policy)
    assert results[0].added_ids == ["#/h"]
    assert results[1].added_ids == []
    assert results[2].added_ids == []
    assert sum(result.added_tokens for result in results) <= 5


def test_no_returned_unit_exceeds_the_per_unit_ceiling(graph):
    hits = [("#/b2", 0.9)]
    policy = ExpansionPolicy(mode="siblings", max_tokens_per_unit=6)
    result = expand(graph, hits, policy)[0]
    assert result.unit.token_count <= 6


def test_a_hit_that_cannot_grow_at_all_still_comes_back(graph):
    policy = ExpansionPolicy(mode="parent", max_added_tokens=0)
    result = expand(graph, [("#/b1", 0.9)], policy)[0]
    assert result.unit.member_ids == ["#/b1"]
    assert result.reason == "hit"


def test_an_auto_merge_group_that_will_not_fit_falls_back_to_bare_hits(graph):
    hits = [("#/b1", 0.9), ("#/b2", 0.8)]
    policy = ExpansionPolicy(mode="auto_merge", min_siblings=2, max_added_tokens=0)
    results = expand(graph, hits, policy)
    assert all(result.reason == "hit" for result in results)
    assert all(len(result.unit.member_ids) == 1 for result in results)


# --- provenance --------------------------------------------------------------

def test_an_expanded_unit_still_carries_every_members_provenance(graph):
    """A unit with no geometry cannot be scored against region gold, so expansion
    must not produce one."""
    result = expand(graph, [("#/b1", 0.9)], ExpansionPolicy(mode="parent"))[0]
    assert len(result.unit.provenance) == len(result.unit.member_ids)


def test_expansion_is_deterministic(graph):
    hits = [("#/b1", 0.9), ("#/b2", 0.8), ("#/q", 0.4)]
    policy = ExpansionPolicy(mode="auto_merge")
    assert expand(graph, hits, policy) == expand(graph, hits, policy)
