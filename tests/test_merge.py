"""Merging ancestors into nodes to make retrievable units."""

import pytest

from document2graph.graph_store import DocumentGraph, GraphSnippet
from document2graph.models import MergePolicy, Provenance
from document2graph.retrieval import (
    approximate_token_count,
    calibrate_min_weight,
    mean_unit_tokens,
    merge_units,
)

DOC = "doc-1"
ROOT = "#/root"


def snippet(ref: str, text: str, sequence_no: int, level_label: str = "body",
            page_no: int = 1, snippet_type: str = "text") -> GraphSnippet:
    return GraphSnippet(
        snippet_id=ref, local_ref=ref, document_id=DOC, sequence_no=sequence_no,
        label="text", snippet_type=snippet_type, level=0, level_label=level_label,
        page_no=page_no, text=text,
        provenance=[Provenance(page_no=page_no, charspan=(0, len(text)))],
        extracted_at="2026-01-01T00:00:00Z",
    )


@pytest.fixture
def graph() -> DocumentGraph:
    """A four-deep chain plus a sibling, with the weights the gates are tested on.

        root  "Praxisempfehlung"
         +- H1   "Therapie des Typ 1 Diabetes"          root->H1  1.0
            +- H2   "Bei Kindern"                        H1->H2   1.0
            |   +- P    "Die Dosis wird angepasst:"      H2->P    0.8
            |       +- B    "zweimal taeglich"           P->B     0.6
            +- P2   "Ein eigenstaendiger Absatz."        H1->P2   0.8
    """
    return DocumentGraph(
        document_id=DOC, filename="doc.pdf", title="Praxisempfehlung", root_id=ROOT,
        snippets=[
            snippet(ROOT, "Praxisempfehlung", 0, level_label="Title"),
            snippet("#/h1", "Therapie des Typ 1 Diabetes", 1, level_label="Heading"),
            snippet("#/h2", "Bei Kindern", 2, level_label="Heading"),
            snippet("#/p", "Die Dosis wird angepasst:", 3),
            snippet("#/b", "zweimal taeglich", 4),
            snippet("#/p2", "Ein eigenstaendiger Absatz.", 5),
        ],
        edges=[
            (ROOT, "#/h1", 1.0),
            ("#/h1", "#/h2", 1.0),
            ("#/h2", "#/p", 0.8),
            ("#/p", "#/b", 0.6),
            ("#/h1", "#/p2", 0.8),
        ],
    )


def unit_by_id(units, unit_id):
    return next(unit for unit in units if unit.unit_id == unit_id)


# --- the unit set -----------------------------------------------------------

def test_every_node_with_text_becomes_exactly_one_unit(graph):
    """A policy that emitted only tree leaves would silently drop the text of any
    paragraph that anchors a list -- here "#/p", which parents the bullet."""
    units = merge_units(graph, MergePolicy(strategy="none"))
    assert [unit.unit_id for unit in units] == [ROOT, "#/h1", "#/h2", "#/p", "#/b", "#/p2"]


def test_a_title_node_root_is_a_unit_but_a_synthetic_one_is_not():
    """Nearly every document roots on its own title block -- a real region of page
    one, with text worth retrieving. Only a document whose title could not be
    resolved gets the placeholder, which sits on no page and can never be scored,
    so the two have to be told apart by what the root *is*, not by it being root."""
    def build(root_type: str) -> DocumentGraph:
        return DocumentGraph(
            document_id=DOC, filename="doc.pdf", title="t", root_id=ROOT,
            snippets=[snippet(ROOT, "A title", 0, level_label="Title",
                              snippet_type=root_type),
                      snippet("#/p", "Body text.", 1)],
            edges=[(ROOT, "#/p", 0.8)],
        )

    real = merge_units(build("text"), MergePolicy(strategy="none"))
    placeholder = merge_units(build("root"), MergePolicy(strategy="none"))
    assert [unit.unit_id for unit in real] == [ROOT, "#/p"]
    assert [unit.unit_id for unit in placeholder] == ["#/p"]


@pytest.mark.parametrize("strategy", ["none", "all_ancestors", "weighted", "subtree"])
@pytest.mark.parametrize("min_tokens", [0, 4, 100])
def test_no_node_loses_its_text(graph, strategy, min_tokens):
    """The floor decides what stands on its own, never what survives.

    ``min_tokens=100`` puts every node in this graph below the floor, which is the
    case that catches a floor implemented as a filter: merging only walks upwards, so
    a node left out of the unit set with nothing to fold it into is simply gone.
    """
    units = merge_units(graph, MergePolicy(strategy=strategy, min_weight=0.0,
                                           min_tokens=min_tokens))
    covered = {member for unit in units for member in unit.member_ids}
    assert {s.snippet_id for s in graph.snippets} - covered == set()


def test_units_come_back_in_reading_order(graph):
    units = merge_units(graph, MergePolicy(strategy="weighted"))
    assert [unit.unit_id for unit in units] == [ROOT, "#/h1", "#/h2", "#/p", "#/b", "#/p2"]


# --- what gets merged -------------------------------------------------------

def test_none_leaves_every_node_bare(graph):
    bullet = unit_by_id(merge_units(graph, MergePolicy(strategy="none")), "#/b")
    assert bullet.text == "zweimal taeglich"
    assert bullet.member_ids == ["#/b"]
    assert not bullet.is_merged


def test_all_ancestors_merges_the_whole_chain_outermost_first(graph):
    bullet = unit_by_id(merge_units(graph, MergePolicy(strategy="all_ancestors")), "#/b")
    assert bullet.member_ids == [ROOT, "#/h1", "#/h2", "#/p", "#/b"]
    assert bullet.text.startswith("Praxisempfehlung")
    assert bullet.text.endswith("zweimal taeglich")


def test_a_gate_stops_the_walk_rather_than_skipping_a_link(graph):
    """The bullet reaches its parent over a 0.6 edge. At a 0.7 threshold that hop
    fails, and the strong edges above it must not be merged over the gap: a unit
    built from a heading and a bullet without the paragraph between them is text
    that never sat together in the document."""
    bullet = unit_by_id(merge_units(graph, MergePolicy(strategy="weighted", min_weight=0.7)), "#/b")
    assert bullet.member_ids == ["#/b"]


def test_a_gate_the_edge_passes_lets_the_walk_continue(graph):
    bullet = unit_by_id(merge_units(graph, MergePolicy(strategy="weighted", min_weight=0.6)), "#/b")
    assert bullet.member_ids == [ROOT, "#/h1", "#/h2", "#/p", "#/b"]


def test_the_threshold_admits_an_edge_of_exactly_that_weight(graph):
    paragraph = unit_by_id(merge_units(graph, MergePolicy(strategy="weighted", min_weight=0.8)), "#/p")
    assert paragraph.member_ids == [ROOT, "#/h1", "#/h2", "#/p"]


def test_min_cumulative_stops_a_long_chain_of_mediocre_edges(graph):
    """Each hop passes min_weight on its own; the product of them does not."""
    policy = MergePolicy(strategy="weighted", min_weight=0.5, min_cumulative=0.5)
    bullet = unit_by_id(merge_units(graph, policy), "#/b")
    # 0.6 alone clears it, 0.6 * 0.8 = 0.48 does not
    assert bullet.member_ids == ["#/p", "#/b"]


def test_merged_edges_record_the_weights_the_decision_rested_on(graph):
    bullet = unit_by_id(merge_units(graph, MergePolicy(strategy="all_ancestors")), "#/b")
    assert [(edge.parent_id, edge.weight) for edge in bullet.merged_edges] == [
        ("#/p", 0.6), ("#/h2", 0.8), ("#/h1", 1.0), (ROOT, 1.0),
    ]
    assert bullet.merged_edges[1].cumulative == pytest.approx(0.48)


# --- the token budget -------------------------------------------------------

def test_the_budget_drops_the_outermost_context_and_keeps_the_nearest(graph):
    policy = MergePolicy(strategy="all_ancestors", max_tokens=12)
    bullet = unit_by_id(merge_units(graph, policy), "#/b")
    assert bullet.token_count <= 12
    assert bullet.member_ids[-1] == "#/b"
    assert ROOT not in bullet.member_ids


def test_a_unit_never_exceeds_the_budget_unless_its_own_node_already_does(graph):
    """The budget is measured on the assembled text, not on the sum of the parts.
    Those differ -- the separator costs tokens, and a subword tokenizer does not
    split a concatenation the way it splits the pieces -- and summing the parts
    let units past max_tokens."""
    for max_tokens in (4, 8, 16, 64):
        policy = MergePolicy(strategy="all_ancestors", max_tokens=max_tokens)
        for unit in merge_units(graph, policy):
            own = approximate_token_count(
                next(s.text for s in graph.snippets if s.snippet_id == unit.unit_id)
            )
            assert unit.token_count <= max_tokens or own > max_tokens


def test_a_node_over_budget_on_its_own_merges_nothing(graph):
    policy = MergePolicy(strategy="all_ancestors", max_tokens=2)
    bullet = unit_by_id(merge_units(graph, policy), "#/b")
    assert bullet.member_ids == ["#/b"]


# --- metadata ---------------------------------------------------------------

def test_the_breadcrumb_is_headings_only_outermost_first(graph):
    bullet = unit_by_id(merge_units(graph, MergePolicy(strategy="none")), "#/b")
    # the paragraph between is body text, so it is not part of the heading path
    assert bullet.breadcrumb == ["Praxisempfehlung", "Therapie des Typ 1 Diabetes", "Bei Kindern"]


def test_the_breadcrumb_stays_out_of_the_text(graph):
    """Prepending the heading path is its own retrieval condition, and one any
    segmentation can be given. Folding it in here would make "structure-aware"
    and "heading-prepending" impossible to tell apart."""
    bullet = unit_by_id(merge_units(graph, MergePolicy(strategy="none")), "#/b")
    assert bullet.text == "zweimal taeglich"
    assert bullet.breadcrumb


def test_provenance_of_every_member_travels_with_a_merged_unit(graph):
    bullet = unit_by_id(merge_units(graph, MergePolicy(strategy="all_ancestors")), "#/b")
    assert len(bullet.provenance) == len(bullet.member_ids)


def test_merging_is_deterministic(graph):
    policy = MergePolicy(strategy="weighted", min_weight=0.6)
    assert merge_units(graph, policy) == merge_units(graph, policy)


# --- subtree ----------------------------------------------------------------

def test_subtree_makes_one_unit_per_heading_holding_its_descendants(graph):
    units = merge_units(graph, MergePolicy(strategy="subtree"))
    assert [unit.unit_id for unit in units] == [ROOT, "#/h1", "#/h2"]
    assert unit_by_id(units, "#/h2").member_ids == ["#/h2", "#/p", "#/b"]
    # nested, so the inner section's text is in the outer unit too
    assert unit_by_id(units, "#/h1").member_ids == ["#/h1", "#/h2", "#/p", "#/b", "#/p2"]


def test_subtree_still_emits_a_node_stranded_on_a_synthetic_root():
    """Only a heading that becomes a unit can carry anything. The placeholder root
    does not, so a node hanging directly off it is covered by nothing -- and 275
    nodes across the test corpus hang directly off a root."""
    graph = DocumentGraph(
        document_id=DOC, filename="doc.pdf", title="t", root_id=ROOT,
        snippets=[snippet(ROOT, "Title", 0, level_label="Root", snippet_type="root",
                          page_no=0),
                  snippet("#/orphan", "Stranded on the root.", 1)],
        edges=[(ROOT, "#/orphan", 0.1)],
    )
    units = merge_units(graph, MergePolicy(strategy="subtree"))
    assert [unit.unit_id for unit in units] == ["#/orphan"]


# --- calibration ------------------------------------------------------------

def test_mean_unit_tokens_rises_as_more_is_merged(graph):
    bare = mean_unit_tokens([graph], MergePolicy(strategy="none"))
    everything = mean_unit_tokens([graph], MergePolicy(strategy="all_ancestors"))
    assert bare < everything


def test_calibration_finds_a_threshold_that_hits_a_reachable_target(graph):
    target = mean_unit_tokens([graph], MergePolicy(strategy="weighted", min_weight=0.8))
    calibration = calibrate_min_weight([graph], MergePolicy(strategy="weighted"), target)
    assert calibration.within(0.02)
    assert calibration.achieved_mean_tokens == pytest.approx(target)


def test_calibration_reports_a_target_it_could_not_reach(graph):
    """The reachable means are a step function of the distinct edge weights, so a
    target between two steps is not a search failure but an unreachable ask. It
    has to be visible, or two conditions get reported as budget-matched when one
    merges half again as much as the other."""
    low = mean_unit_tokens([graph], MergePolicy(strategy="none"))
    high = mean_unit_tokens([graph], MergePolicy(strategy="all_ancestors"))
    calibration = calibrate_min_weight([graph], MergePolicy(strategy="weighted"),
                                       (low + high) / 2)
    assert not calibration.within(0.001)
    assert calibration.relative_error != 0.0


def test_calibration_tries_every_distinct_weight_when_there_are_few(graph):
    calibration = calibrate_min_weight([graph], MergePolicy(strategy="weighted"), 20.0)
    assert calibration.exhaustive
    # three distinct weights in the fixture (0.6, 0.8, 1.0), plus 0.0 and one above
    assert calibration.candidates_evaluated == 5


def test_calibration_rejects_a_target_that_cannot_be_a_size(graph):
    with pytest.raises(ValueError):
        calibrate_min_weight([graph], MergePolicy(strategy="weighted"), 0.0)


# --------------------------------------------------------------------------- #
# regressions
# --------------------------------------------------------------------------- #
@pytest.fixture
def overflowing_section() -> DocumentGraph:
    """One heading with three body paragraphs, too much text to merge as one.

        root "Doc"
         +- H "Heading"
             +- P1 / P2 / P3, ten words each
    """
    body = " ".join(f"wort{i}" for i in range(10))
    return DocumentGraph(
        document_id=DOC, filename="doc.pdf", title="Doc", root_id=ROOT,
        snippets=[
            snippet(ROOT, "Doc", 0, level_label="Title"),
            snippet("#/h", "Heading", 1, level_label="Heading"),
            snippet("#/p1", body, 2),
            snippet("#/p2", body, 3),
            snippet("#/p3", body, 4),
        ],
        edges=[(ROOT, "#/h", 1.0), ("#/h", "#/p1", 0.8),
               ("#/h", "#/p2", 0.8), ("#/h", "#/p3", 0.8)],
    )


def test_a_truncated_subtree_leaves_its_tail_retrievable(overflowing_section):
    """``max_tokens`` may split a section; it must not delete the rest of it.

    The heading stops merging at the budget. Whatever it did not reach is
    carried by no unit, so it has to stand alone -- otherwise its text *and its
    geometry* are absent from the corpus, and a retrieval benchmark built on
    these units is scoring a document it cannot see all of.
    """
    units = merge_units(overflowing_section, MergePolicy(strategy="subtree", max_tokens=15))

    carried = {member_id for unit in units for member_id in unit.member_ids}
    text_bearing = {s.snippet_id for s in overflowing_section.snippets if s.text.strip()}
    assert text_bearing <= carried, f"dropped from the corpus: {sorted(text_bearing - carried)}"

    # and the tail is reachable on its own terms, not only as somebody's member
    assert {"#/p2", "#/p3"} <= {unit.unit_id for unit in units}


def test_every_unit_stays_within_the_token_budget(overflowing_section):
    """The tail is rescued as separate units, not by overflowing the budget."""
    policy = MergePolicy(strategy="subtree", max_tokens=15)
    for unit in merge_units(overflowing_section, policy):
        assert approximate_token_count(unit.text) <= policy.max_tokens


@pytest.fixture
def labelled_chart() -> DocumentGraph:
    """A textless picture with a data label under it -- the chart-page shape.

    The picture sits on page 9 and everything else on page 1, so provenance
    that leaked from the picture is visible as a page number.
    """
    picture = snippet("#/pic", "", 1, page_no=9, snippet_type="image")
    picture.provenance = [Provenance(page_no=9, charspan=(0, 0))]
    return DocumentGraph(
        document_id=DOC, filename="doc.pdf", title="Doc", root_id=ROOT,
        snippets=[
            snippet(ROOT, "Doc", 0, level_label="Title"),
            picture,
            snippet("#/label", "38", 2),
        ],
        edges=[(ROOT, "#/pic", 0.9), ("#/pic", "#/label", 0.8)],
    )


def test_a_merged_ancestor_with_no_text_contributes_no_geometry(labelled_chart):
    """A unit reading "38" must not claim the box of the chart it sits in.

    Provenance is what a unit *returned*, and a textless picture returns
    nothing. Claiming it would let this unit cover a figure's gold region under
    an overlap metric while the text a retriever indexed is one number.
    """
    unit = unit_by_id(merge_units(labelled_chart, MergePolicy(strategy="weighted")), "#/label")

    assert "#/pic" in unit.member_ids  # it is still an ancestor of the unit
    assert unit.text.endswith("38")  # the title merges too, the textless picture does not
    assert 9 not in [p.page_no for p in unit.provenance], (
        "the textless picture on page 9 leaked its geometry into a text unit"
    )
    assert unit.provenance, "the unit still has to be placeable on a page"


# --------------------------------------------------------------------------- #
# the min_tokens floor
# --------------------------------------------------------------------------- #
def test_a_node_below_the_floor_is_not_a_unit_of_its_own(graph):
    """"zweimal taeglich" is two words: nothing in it for a query to match."""
    units = merge_units(graph, MergePolicy(strategy="none", min_tokens=5))
    assert "#/b" not in {unit.unit_id for unit in units}


def test_a_node_below_the_floor_joins_its_nearest_standing_ancestor(graph):
    """The bullet reads under the paragraph that introduces it, in that order.

    Nearest, not any: folding it into the heading two levels up would put it next to
    text it never sat next to on the page.
    """
    paragraph = unit_by_id(merge_units(graph, MergePolicy(strategy="none", min_tokens=5)), "#/p")

    assert paragraph.member_ids == ["#/p", "#/b"]
    assert paragraph.text == "Die Dosis wird angepasst:\n\nzweimal taeglich"


def test_a_node_below_the_floor_is_folded_into_exactly_one_unit(graph):
    units = merge_units(graph, MergePolicy(strategy="none", min_tokens=5))
    holders = [unit.unit_id for unit in units if "#/b" in unit.member_ids]
    assert holders == ["#/p"]


def test_a_chain_of_small_nodes_folds_onto_the_one_that_still_stands(graph):
    """A host has to be a unit itself, or the fold lands on something unretrievable.

    A floor above every node in the document leaves only the title, which stands
    because there is nothing above it to fold *it* into. Everything else belongs
    there -- not to the nearest ancestor, which is itself folded away.
    """
    units = merge_units(graph, MergePolicy(strategy="none", min_tokens=7))

    assert [unit.unit_id for unit in units] == [ROOT]
    assert units[0].member_ids == [ROOT, "#/h1", "#/h2", "#/p", "#/b", "#/p2"]


def test_a_small_node_with_nowhere_to_go_keeps_its_own_unit():
    """Better a unit too small than a piece of the document nothing can return.

    The synthetic root is not a unit, so a fragment stranded on it has no host; the
    floor has to leave it alone rather than drop it.
    """
    stranded = DocumentGraph(
        document_id=DOC, filename="doc.pdf", title="Doc", root_id=ROOT,
        snippets=[snippet(ROOT, "Doc", 0, level_label="Title", snippet_type="root"),
                  snippet("#/frag", "Fortsetzung", 1)],
        edges=[(ROOT, "#/frag", 0.1)],
    )
    units = merge_units(stranded, MergePolicy(strategy="none", min_tokens=4))

    assert [unit.unit_id for unit in units] == ["#/frag"]


def test_a_folded_child_is_charged_to_the_budget_before_any_ancestor(graph):
    """The fold is not droppable, so it is what an ancestor has to fit around."""
    policy = MergePolicy(strategy="all_ancestors", min_tokens=5, max_tokens=12)
    paragraph = unit_by_id(merge_units(graph, policy), "#/p")

    # the bullet is in; the heading above is what did not fit around it
    assert paragraph.member_ids == ["#/p", "#/b"]
    assert approximate_token_count(paragraph.text) <= policy.max_tokens


def test_the_floor_does_not_disturb_a_graph_that_clears_it(graph):
    """min_tokens=0 is the default and has to change nothing at all."""
    policy = dict(strategy="weighted", min_weight=0.5)
    assert ([u.model_dump() for u in merge_units(graph, MergePolicy(**policy))]
            == [u.model_dump() for u in merge_units(graph, MergePolicy(**policy, min_tokens=0))])
