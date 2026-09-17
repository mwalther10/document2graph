"""Supply measures: antecedent coverage, complementarity, the structural prior,
and the composite that floors the first two at the third."""

import pytest

from document2graph.edge_weights import (
    EdgeContext,
    MetricDeps,
    antecedent_coverage,
    complementarity,
    components,
    composite_supply,
    compute_supply,
    idf_table,
    structural_prior_weights,
    supply_relation,
)
from document2graph.edge_weights.antecedent import coverage_support, edge_coverage
from document2graph.edge_weights.complementarity import edge_complementarity, novel_mass
from document2graph.edge_weights.context import (
    KIND_IMAGE,
    KIND_TABLE,
    RELATION_LIST_ITEM,
    RELATION_REFERENCE,
    RELATION_ROOT,
    RELATION_SECTION,
    RELATION_TEXT,
    Unit,
)
from document2graph.edge_weights.reference_density import unresolved_pointers
from document2graph.models import (
    AntecedentCoverageConfig,
    ComplementarityConfig,
    EdgeWeightConfig,
    NeedConfig,
    ReferenceDensityConfig,
    StructuralPriorConfig,
    SupplyConfig,
)

DE = ReferenceDensityConfig(language="de", use_spacy=False,
                            pointer_classes=["anaphor", "demonstrative", "definite"])
COVER = AntecedentCoverageConfig(pointers=DE, idf_weighted=False)


def ctx_of(edges, texts, relations=None, units=None):
    ctx = EdgeContext.from_texts(edges, texts, relations)
    for node_id, patch in (units or {}).items():
        ctx.units[node_id] = Unit(node_id=node_id, text=texts.get(node_id, ""), **patch)
    return ctx


# --- the derived relation taxonomy ---------------------------------------

def test_text_relation_splits_by_what_parents_it():
    texts = {"h": "Therapie", "p": "Ein Satz.", "b": "ein Punkt"}
    heading = {"level_label": "Heading"}
    ctx = ctx_of([("h", "p"), ("p", "b")], texts,
                 {("h", "p"): RELATION_TEXT, ("p", "b"): RELATION_TEXT},
                 units={"h": heading})
    assert supply_relation(ctx, ("h", "p")) == "heading_text"
    assert supply_relation(ctx, ("p", "b")) == "text_text"


def test_list_relation_splits_by_what_anchors_it():
    texts = {"h": "Therapie", "p": "Folgendes gilt:", "b1": "Punkt", "b2": "Punkt"}
    ctx = ctx_of([("h", "b1"), ("p", "b2")], texts,
                 {("h", "b1"): RELATION_LIST_ITEM, ("p", "b2"): RELATION_LIST_ITEM},
                 units={"h": {"level_label": "Heading"},
                        "b1": {"is_grouped": True}, "b2": {"is_grouped": True}})
    assert supply_relation(ctx, ("h", "b1")) == "heading_list"
    assert supply_relation(ctx, ("p", "b2")) == "text_list"


def test_media_splits_table_from_figure():
    texts = {"p": "Siehe Tabelle.", "t": "| a | b |", "f": "Abb.1"}
    ctx = ctx_of([("p", "t"), ("p", "f")], texts,
                 units={"t": {"kind": KIND_TABLE}, "f": {"kind": KIND_IMAGE}})
    assert supply_relation(ctx, ("p", "t")) == "media_table"
    assert supply_relation(ctx, ("p", "f")) == "media_figure"


def test_construction_only_relations_are_read_from_the_context():
    # reference/root cannot be recovered from the two units: they are facts
    # about how the edge was made
    texts = {"a": "Text", "b": "Anderer Text", "r": "Titel"}
    ctx = ctx_of([("a", "b"), ("r", "b")], texts,
                 {("a", "b"): RELATION_REFERENCE, ("r", "b"): RELATION_ROOT})
    assert supply_relation(ctx, ("a", "b")) == "reference"
    assert supply_relation(ctx, ("r", "b")) == "root"


def test_relation_is_derivable_for_an_edge_that_never_existed():
    """The auto-merge ranking scores child->ancestor edges with no stored relation."""
    texts = {"h": "Therapie", "b": "ein Punkt"}
    ctx = ctx_of([], texts, units={"h": {"level_label": "Heading"},
                                   "b": {"is_grouped": True}})
    ctx.edges = [("h", "b")]  # a virtual edge: absent from ctx.relations
    assert ("h", "b") not in ctx.relations
    assert supply_relation(ctx, ("h", "b")) == "heading_list"


# --- antecedent coverage --------------------------------------------------

def test_coverage_is_the_share_of_dangling_pointers_the_parent_answers():
    child = "Die Dosis und die Kontrolle sind entscheidend."
    dangling = unresolved_pointers(child, DE)
    assert {p.key.lower() for p in dangling} == {"dosis", "kontrolle"}
    # a parent naming one of the two heads answers half of them
    assert edge_coverage(child, "Dosis bei Nierenversagen", COVER) == pytest.approx(0.5)
    assert edge_coverage(child, "Dosis und Kontrolle", COVER) == pytest.approx(1.0)
    assert edge_coverage(child, "Ein anderes Thema", COVER) == pytest.approx(0.0)


def test_coverage_is_a_ratio_so_a_long_child_is_not_penalised():
    short = "Die Dosis ist wichtig."
    # filler that raises no pointer of its own: no article, no demonstrative
    long = short + " " + "Messwerte werden regelmaessig erhoben und dokumentiert. " * 4
    parent = "Dosis"
    assert unresolved_pointers(short, DE) and \
           len(unresolved_pointers(long, DE)) == len(unresolved_pointers(short, DE))
    assert edge_coverage(short, parent, COVER) == edge_coverage(long, parent, COVER)


def test_need_times_coverage_is_the_uncovered_share_of_all_pointers():
    """The identity that makes the number readable: need*supply = covered/total."""
    from document2graph.edge_weights.reference_density import analyze_unit_cached, unit_need
    child = "Die Dosis und die Kontrolle sind entscheidend."
    parent = "Dosis bei Nierenversagen"
    total = len(analyze_unit_cached(child, DE))
    covered = sum(1 for p in unresolved_pointers(child, DE)
                  if p.key and p.key.lower().startswith("dosis"))
    product = unit_need(child, DE) * edge_coverage(child, parent, COVER)
    assert product == pytest.approx(covered / total)


def test_a_child_with_nothing_dangling_takes_the_configured_default():
    config = AntecedentCoverageConfig(pointers=DE, default_when_no_pointers=0.3)
    assert coverage_support(
        ctx_of([("p", "c")], {"p": "Titel", "c": "Sieben Punkte."}), config
    )[("p", "c")] == 0
    assert edge_coverage("Sieben Punkte.", "Titel", config) == pytest.approx(0.3)


def test_acronyms_are_out_of_the_denominator_by_default():
    """Their expansion sits in a sibling or an appendix, never in the ancestor."""
    child = "Der HbA1c-Wert und die CGM-Messung sind relevant."
    default = AntecedentCoverageConfig()
    with_acronyms = AntecedentCoverageConfig(
        pointers=ReferenceDensityConfig(
            language="de", use_spacy=False,
            pointer_classes=["anaphor", "demonstrative", "definite", "acronym"]))
    assert not any(p.kind == "acronym" for p in unresolved_pointers(child, default.pointers))
    assert any(p.kind == "acronym" for p in unresolved_pointers(child, with_acronyms.pointers))


def test_idf_weighting_discounts_a_domain_ubiquitous_head():
    child = "Die Patienten und die Dosisanpassung sind zu beachten."
    # "patient" everywhere in the corpus, "dosisanpassung" in one document
    corpus = ["Patienten " + str(i) for i in range(20)] + ["Dosisanpassung bei Niereninsuffizienz"]
    idf = idf_table(corpus)
    weighted = AntecedentCoverageConfig(pointers=DE, idf_weighted=True)
    # a parent supplying only the rare head beats one supplying only the common head
    rare = edge_coverage(child, "Dosisanpassung", weighted, idf=idf)
    common = edge_coverage(child, "Patienten", weighted, idf=idf)
    assert rare > common
    # unweighted, the two are indistinguishable
    flat = AntecedentCoverageConfig(pointers=DE, idf_weighted=False)
    assert edge_coverage(child, "Dosisanpassung", flat) == edge_coverage(child, "Patienten", flat)


# --- complementarity ------------------------------------------------------

def test_a_parent_that_restates_the_child_adds_nothing():
    config = ComplementarityConfig(idf_source="document")
    idf = idf_table(["Blutzucker messen", "andere Inhalte", "noch ein Text"])
    child = "Blutzucker messen"
    assert edge_complementarity(child, child, config, idf) == pytest.approx(0.0)
    assert edge_complementarity("Blutzucker messen bei Schwangeren", child, config, idf) > 0


def test_stemming_collapses_a_german_compound_onto_its_head():
    config = ComplementarityConfig(stem_terms=True)
    unstemmed = ComplementarityConfig(stem_terms=False)
    idf = idf_table(["Blutzuckerspiegel", "Blutzucker", "anderes"], stem=False)
    # "Blutzuckerspiegel" in the parent is not new content when the child says
    # "Blutzucker" -- only the unstemmed reading thinks it is
    stemmed_novel, _ = novel_mass("Blutzuckerspiegel", "Blutzucker", config, idf_table(
        ["Blutzuckerspiegel", "Blutzucker", "anderes"]))
    raw_novel, _ = novel_mass("Blutzuckerspiegel", "Blutzucker", unstemmed, idf)
    assert stemmed_novel == 0.0
    assert raw_novel > 0.0


def test_top_k_stops_the_score_from_rising_with_parent_length():
    capped = ComplementarityConfig(top_k_novel_terms=5, idf_source="document")
    uncapped = ComplementarityConfig(top_k_novel_terms=None, idf_source="document")
    child = "Die Dosis"
    short = "Nierenversagen"
    long = short + " " + " ".join(f"zusatzwort{i}" for i in range(60))
    idf = idf_table([child, short, long, "weiterer Text"])
    assert edge_complementarity(long, child, uncapped, idf) > \
           edge_complementarity(short, child, uncapped, idf)
    # capped, a long parent no longer outscores a short one merely by being long
    assert edge_complementarity(long, child, capped, idf) == pytest.approx(
        edge_complementarity(short + " " + " ".join(f"zusatzwort{i}" for i in range(4)),
                             child, capped, idf), abs=0.15)


def test_a_table_child_floors_out_because_its_own_mass_swamps_the_parent():
    config = ComplementarityConfig(idf_source="document")
    table = " ".join(f"zelle{i} wert{i}" for i in range(120))
    idf = idf_table([table, "Ziele", "anderes"])
    assert edge_complementarity("Ziele", table, config, idf) < 0.1


def test_corpus_idf_is_used_when_supplied_and_falls_back_when_not():
    ctx = ctx_of([("p", "c")], {"p": "Nierenversagen", "c": "Die Dosis", "x": "Fuellt"})
    corpus = idf_table(["Nierenversagen"] + ["haeufiges Wort"] * 30)
    with_corpus = complementarity(ctx, ComplementarityConfig(idf_source="corpus"), corpus)
    without = complementarity(ctx, ComplementarityConfig(idf_source="corpus"), None)
    assert 0.0 <= with_corpus[("p", "c")] <= 1.0
    assert 0.0 <= without[("p", "c")] <= 1.0  # fell back to the document's own nodes


# --- structural prior and the composite -----------------------------------

def test_prior_comes_from_the_relation_and_is_configurable():
    texts = {"h": "Therapie", "b": "Punkt"}
    ctx = ctx_of([("h", "b")], texts, {("h", "b"): RELATION_LIST_ITEM},
                 units={"h": {"level_label": "Heading"}, "b": {"is_grouped": True}})
    assert structural_prior_weights(ctx, StructuralPriorConfig())[("h", "b")] == pytest.approx(0.80)
    custom = StructuralPriorConfig(priors={"heading_list": 0.2})
    assert structural_prior_weights(ctx, custom)[("h", "b")] == pytest.approx(0.2)
    # a relation absent from a replaced table falls back to `default`
    assert structural_prior_weights(
        ctx, StructuralPriorConfig(priors={}, default=0.33))[("h", "b")] == pytest.approx(0.33)


def test_priors_outside_the_unit_range_are_rejected():
    with pytest.raises(ValueError):
        StructuralPriorConfig(priors={"section": 1.4})


def test_the_prior_is_a_floor_and_never_a_ceiling():
    texts = {"p": "Nierenversagen Dosisanpassung Schwangerschaft", "c": "Die Dosis"}
    ctx = ctx_of([("p", "c")], texts, {("p", "c"): RELATION_TEXT})
    low = SupplyConfig(structural_prior=StructuralPriorConfig(priors={"text_text": 0.0}),
                       antecedent=COVER)
    high = SupplyConfig(structural_prior=StructuralPriorConfig(priors={"text_text": 0.95}),
                        antecedent=COVER)
    measured = composite_supply(ctx, low)[("p", "c")]
    floored = composite_supply(ctx, high)[("p", "c")]
    assert floored == pytest.approx(max(measured, 0.95))
    assert floored >= measured


def test_composite_takes_the_max_so_one_silent_measure_does_not_halve_it():
    texts = {"p": "Nierenversagen Dosisanpassung", "c": "Die Dosis ist wichtig."}
    ctx = ctx_of([("p", "c")], texts, {("p", "c"): RELATION_TEXT})
    no_floor = StructuralPriorConfig(priors={"text_text": 0.0})
    as_max = composite_supply(ctx, SupplyConfig(combine="max", structural_prior=no_floor,
                                                antecedent=COVER))[("p", "c")]
    as_mean = composite_supply(ctx, SupplyConfig(combine="mean", structural_prior=no_floor,
                                                 antecedent=COVER))[("p", "c")]
    parts = components(ctx, SupplyConfig(structural_prior=no_floor, antecedent=COVER))[("p", "c")]
    assert as_max == pytest.approx(max(parts["antecedent_coverage"], parts["complementarity"]))
    assert as_mean <= as_max


def test_floor_can_be_switched_off_for_the_ablation():
    texts = {"p": "Therapie", "c": "Punkt"}
    ctx = ctx_of([("p", "c")], texts, {("p", "c"): RELATION_TEXT})
    off = composite_supply(ctx, SupplyConfig(floor="none", antecedent=COVER))[("p", "c")]
    on = composite_supply(ctx, SupplyConfig(floor="prior", antecedent=COVER))[("p", "c")]
    assert off <= on
    assert on >= 0.40  # the text_text prior


def test_components_report_whether_the_prior_or_the_evidence_produced_the_score():
    # a child that raises no pointer and shares every term with its parent gives
    # both measured components nothing, so the prior is what produces the score
    texts = {"p": "Ziele", "c": "Ziele"}
    ctx = ctx_of([("p", "c")], texts, {("p", "c"): RELATION_TEXT})
    parts = components(ctx, SupplyConfig(antecedent=COVER))[("p", "c")]
    assert parts["antecedent_coverage"] == 0.0 and parts["complementarity"] == 0.0
    assert parts["floor_bound"] == 1.0
    assert parts["supply"] == pytest.approx(parts["structural_prior"])


def test_a_very_short_child_scores_high_on_complementarity_by_construction():
    """Not a defect, but the reading has to be held to: the score is the share of
    the merged unit's informative mass the parent contributes, and against a
    three-word bullet a heading really does contribute most of it."""
    config = ComplementarityConfig(idf_source="document")
    idf = idf_table(["Therapie bei Niereninsuffizienz", "Schwere Leberinsuffizienz", "x y z"])
    assert edge_complementarity("Therapie bei Niereninsuffizienz",
                                "Schwere Leberinsuffizienz", config, idf) > 0.5


def test_every_supply_metric_stays_in_the_unit_range():
    texts = {"h": "Therapie des Typ-2-Diabetes", "p": "Die Dosis soll angepasst werden.",
             "b": "bei Niereninsuffizienz", "x": "Ein unabhaengiger Abschnitt."}
    ctx = ctx_of([("h", "p"), ("p", "b")], texts,
                 {("h", "p"): RELATION_TEXT, ("p", "b"): RELATION_LIST_ITEM})
    for metric in ("uniform", "structural", "antecedent_coverage",
                   "complementarity", "composite"):
        supply = compute_supply(ctx, EdgeWeightConfig(),
                                SupplyConfig(metric=metric, antecedent=COVER), MetricDeps())
        assert all(0.0 <= v <= 1.0 for v in supply.values()), metric


def test_composite_is_the_default_and_needs_no_model():
    config = EdgeWeightConfig(metric="need_supply",
                              need=NeedConfig(estimator="reference_density"))
    assert config.supply.metric == "composite"
    texts = {"h": "Therapie bei Niereninsuffizienz", "b": "Die Dosis anpassen."}
    ctx = ctx_of([("h", "b")], texts, {("h", "b"): RELATION_TEXT})
    from document2graph.edge_weights import compute_edge_weights
    weights = compute_edge_weights(ctx, config)  # no encoder, no scorer injected
    assert all(0.0 <= v <= 1.0 for v in weights.values())


# --- the hyphen fix and the prose-parent restriction ----------------------

def test_a_hyphenated_compound_matches_the_same_compound_in_the_parent():
    """stem("HbA1c-Messung") ran into the hyphen and could never match a parent
    tokenized on word characters, so an identical compound scored zero."""
    from document2graph.edge_weights.antecedent import key_stems
    child = "Die HbA1c-Messung ist entscheidend."
    assert key_stems("HbA1c-Messung") == ["hba1c", "messun"]
    assert edge_coverage(child, "HbA1c-Messung im Labor", COVER) == pytest.approx(1.0)
    # and it does not loosen the test: half the compound is not the entity
    assert edge_coverage(child, "Eine Messung im Labor", COVER) == pytest.approx(0.0)


def test_numerals_in_a_compound_are_not_required_of_the_parent():
    from document2graph.edge_weights.antecedent import key_stems
    assert key_stems("Typ-1-Diabetes") == ["typ", "diabet"]


def test_coverage_says_nothing_about_a_heading_parent_by_default():
    texts = {"h": "Praeanalytische Stabilitaet", "p": "Die Lagerung ist entscheidend."}
    ctx = ctx_of([("h", "p")], texts, units={"h": {"level_label": "Heading"}})
    assert coverage_support(ctx, AntecedentCoverageConfig())[("h", "p")] == 0
    # the child does raise a pointer; it is the heading parent that is skipped
    assert coverage_support(ctx, AntecedentCoverageConfig(prose_parents_only=False))[("h", "p")] > 0


# --- counterfactual re-centring -------------------------------------------

def test_controls_are_never_the_child_s_own_ancestors():
    from document2graph.edge_weights import control_parents
    from document2graph.models import CounterfactualConfig
    texts = {"a": "Wurzel Text", "b": "Mittlerer Text", "c": "Kind Text", "d": "Anderer Text",
             "e": "Noch ein Text"}
    ctx = ctx_of([("a", "b"), ("b", "c"), ("d", "e")], texts)
    controls = control_parents(ctx, ("b", "c"),
                               CounterfactualConfig(match_relation=False, match_length=False))
    assert "a" not in controls and "b" not in controls and "c" not in controls
    assert "d" in controls


def test_recentring_cancels_a_baseline_every_parent_would_have_scored():
    from document2graph.edge_weights import recentre
    from document2graph.models import CounterfactualConfig
    texts = {"p": "Alpha", "q": "Beta", "r": "Gamma", "c": "Kind", "x": "Delta"}
    ctx = ctx_of([("p", "c"), ("q", "x"), ("r", "x")], texts)
    flat = lambda context: {edge: 0.7 for edge in context.edges}
    gaps = recentre(ctx, CounterfactualConfig(match_relation=False, match_length=False), flat)
    assert gaps[("p", "c")] == pytest.approx(0.0)  # no parent is special here


def test_recentring_keeps_what_is_specific_to_the_pair():
    from document2graph.edge_weights import recentre
    from document2graph.models import CounterfactualConfig
    texts = {"p": "Alpha", "q": "Beta", "r": "Gamma", "c": "Kind", "x": "Delta"}
    ctx = ctx_of([("p", "c"), ("q", "x"), ("r", "x")], texts)
    special = lambda context: {edge: (0.9 if edge[0] == "p" else 0.4) for edge in context.edges}
    gaps = recentre(ctx, CounterfactualConfig(match_relation=False, match_length=False), special)
    assert gaps[("p", "c")] == pytest.approx(0.5)


def test_gap_is_clipped_at_zero_when_used_as_supply():
    from document2graph.edge_weights import as_supply
    assert as_supply({("a", "b"): -0.3, ("c", "d"): 0.2}) == {("a", "b"): 0.0, ("c", "d"): 0.2}


def test_counterfactual_is_off_by_default_and_changes_the_components_when_on():
    from document2graph.edge_weights import measured_components
    from document2graph.models import CounterfactualConfig
    assert SupplyConfig().counterfactual.enabled is False
    texts = {"p": "Nierenversagen Dosisanpassung", "q": "Ein anderer Abschnitt hier",
             "c": "Die Dosis", "x": "Weiterer Text"}
    ctx = ctx_of([("p", "c"), ("q", "x")], texts)
    off = measured_components(ctx, SupplyConfig(antecedent=COVER), MetricDeps())
    on = measured_components(
        ctx, SupplyConfig(antecedent=COVER,
                          counterfactual=CounterfactualConfig(enabled=True, match_relation=False)),
        MetricDeps())
    assert all(0.0 <= v <= 1.0 for part in on for v in part.values())
    assert on != off
