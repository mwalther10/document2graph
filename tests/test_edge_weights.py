import numpy as np
import pytest

from document2graph.models import (
    EdgeWeightConfig,
    NeedConfig,
    NeedSupplyConfig,
    ReferenceDensityConfig,
    SimilarityConfig,
    SupplyConfig,
    SurprisalConfig,
    SyntacticConfig,
)
from document2graph.edge_weights import (
    EdgeContext,
    MetricDeps,
    apply_edge_weights,
    available_metrics,
    compute_edge_weights,
    compute_metric,
    compute_need,
    compute_supply,
    content_similarity,
    evidence_components,
    inverted_similarity,
    register_metric,
)
from document2graph.edge_weights.context import KIND_TABLE, Unit
from document2graph.edge_weights.lexical import bm25_similarity
from document2graph.edge_weights.semantic import embedding_similarity
from document2graph.edge_weights.reference_density import analyze_unit, unit_need
from document2graph.edge_weights.surprisal import delta_nll, normalize, surprisal_need
from document2graph.edge_weights.syntactic import edge_features, score_features

# corpus large enough that terms shared by only two nodes keep a positive IDF
TEXTS = {
    "root": "Test Document",
    "n1": "glucose metabolism in diabetic patients",
    "n2": "insulin therapy for glucose metabolism disorders",
    "n3": "unrelated section about publication venue",
    "n4": "figure caption showing enzyme kinetics",
    "n5": "appendix with abbreviations",
    "n6": "acknowledgements and funding statement",
}
EDGES = [("n1", "n2"), ("n3", "n4")]


def ctx_of(edges=EDGES, texts=TEXTS, relations=None) -> EdgeContext:
    return EdgeContext.from_texts(edges, texts, relations)


def fake_encode(batch: list[str]) -> np.ndarray:
    # deterministic 3-dim embeddings keyed by text
    vectors = {
        TEXTS["n1"]: [1.0, 0.0, 0.0],
        TEXTS["n2"]: [1.0, 1.0, 0.0],   # cos(n1, n2) = 1/sqrt(2)
        TEXTS["n3"]: [0.0, 1.0, 0.0],
        TEXTS["n4"]: [0.0, 0.0, 1.0],   # cos(n3, n4) = 0
    }
    return np.array([vectors.get(text, [0.0, 0.0, 0.0]) for text in batch])


# --- similarity -----------------------------------------------------------

def test_bm25_similarity_is_normalized_to_the_strongest_edge():
    similarity = bm25_similarity(ctx_of())
    assert set(similarity) == set(EDGES)
    assert all(0.0 <= value <= 1.0 for value in similarity.values())
    # n1/n2 share "glucose metabolism" -> the most similar edge pins to 1
    assert similarity[("n1", "n2")] == 1.0
    # n3/n4 share no terms -> BM25 clips to 0
    assert similarity[("n3", "n4")] == 0.0


def test_bm25_symmetric_modes_are_direction_invariant():
    for scoring in ("symmetric_mean", "symmetric_max"):
        similarity = bm25_similarity(ctx_of([("n1", "n2"), ("n2", "n1")]), scoring=scoring)
        assert similarity[("n1", "n2")] == pytest.approx(similarity[("n2", "n1")])
    asymmetric = bm25_similarity(ctx_of([("n1", "n2"), ("n2", "n1")]), scoring="child_query")
    assert asymmetric[("n1", "n2")] != pytest.approx(asymmetric[("n2", "n1")])


def test_bm25_handles_empty_and_unknown_nodes():
    similarity = bm25_similarity(ctx_of([("missing", "n1"), ("n1", "n2")]))
    assert similarity[("missing", "n1")] == 0.0
    all_empty = bm25_similarity(ctx_of([("a", "b")], {"a": "", "b": ""}))
    assert all_empty[("a", "b")] == 0.0


def test_embedding_similarity_is_clipped_cosine():
    similarity = embedding_similarity(ctx_of(), model_name="unused", encode=fake_encode)
    assert similarity[("n1", "n2")] == pytest.approx(1.0 / np.sqrt(2))
    assert similarity[("n3", "n4")] == 0.0  # orthogonal embeddings


def test_blend_interpolates_between_lexical_and_semantic():
    lexical = bm25_similarity(ctx_of())
    semantic = embedding_similarity(ctx_of(), model_name="unused", encode=fake_encode)
    for alpha in (0.0, 0.3, 1.0):
        config = SimilarityConfig(backend="blend", alpha=alpha, embedding_model="unused")
        blended = content_similarity(ctx_of(), config, encode=fake_encode)
        for edge in EDGES:
            expected = alpha * semantic[edge] + (1 - alpha) * lexical[edge]
            assert blended[edge] == pytest.approx(expected)


def test_inverted_similarity_is_one_minus_similarity():
    config = SimilarityConfig(backend="bm25")
    similarity = content_similarity(ctx_of(), config)
    inverted = inverted_similarity(ctx_of(), config)
    for edge in EDGES:
        assert inverted[edge] == pytest.approx(1.0 - similarity[edge])


# --- need: unresolved reference density -----------------------------------

def test_reference_density_counts_only_unresolved_pointers():
    config = ReferenceDensityConfig()
    # "Die Therapie" is introduced as "Eine Therapie" first -> resolved
    assert unit_need("Eine Therapie mit Metformin wurde begonnen. Die Therapie war wirksam.",
                     config) == 0.0
    # a pronoun and a demonstrative with nothing to point at inside the unit
    assert unit_need("Sie sollte bei diesen Patienten auf 7% gesenkt werden.", config) == 1.0


def test_reference_density_is_a_ratio_not_a_count():
    config = ReferenceDensityConfig()
    short = "Die Therapie ist wirksam."
    padded = short + " " + "Wasser besteht aus Wasserstoff und Sauerstoff. " * 5
    # length alone must not make a unit look self-sufficient
    assert unit_need(padded, config) == pytest.approx(unit_need(short, config))


def test_acronym_is_resolved_by_an_in_unit_expansion():
    config = ReferenceDensityConfig(pointer_classes=["acronym"])
    assert unit_need("Die GFR war normal.", config) == 1.0
    assert unit_need("Die glomeruläre Filtrationsrate (GFR) war normal.", config) == 0.0


def test_units_without_pointers_take_the_configured_default():
    assert unit_need("Wasser besteht aus Wasserstoff und Sauerstoff.",
                     ReferenceDensityConfig()) == 0.0
    assert unit_need("Wasser besteht aus Wasserstoff und Sauerstoff.",
                     ReferenceDensityConfig(default_when_no_pointers=0.5)) == 0.5


def test_pointer_classes_can_be_ablated():
    text = "Sie sollte bei diesen Patienten gesenkt werden."
    kinds = {pointer.kind for pointer in analyze_unit(text, ReferenceDensityConfig())}
    assert {"anaphor", "demonstrative"} <= kinds
    only_anaphors = analyze_unit(text, ReferenceDensityConfig(pointer_classes=["anaphor"]))
    assert {pointer.kind for pointer in only_anaphors} == {"anaphor"}


def test_reference_density_broadcasts_to_every_in_edge():
    needs = compute_need(
        ctx_of([("a", "c"), ("b", "c")], {"a": "A", "b": "B", "c": "Sie ist wirksam."}),
        NeedConfig(estimator="reference_density"),
    )
    assert needs[("a", "c")] == needs[("b", "c")]


# --- need: syntactic self-sufficiency -------------------------------------

def test_bullet_under_a_colon_scores_higher_than_a_full_sentence():
    config = SyntacticConfig()
    bullet = score_features(
        edge_features("senken auf 7% bei Patienten unter 65", "Zielwerte:", config),
        config.feature_weights)
    sentence = score_features(
        edge_features("Der Blutzucker soll vor dem Essen gemessen werden.",
                      "Diabetes mellitus Typ 2", config),
        config.feature_weights)
    assert bullet > 0.5 > sentence
    assert sentence == 0.0


def test_colon_is_a_property_of_the_edge_not_the_child():
    config = SyntacticConfig()
    child = "Metformin 500 mg zweimal täglich"
    with_colon = edge_features(child, "Therapieempfehlung:", config)
    without_colon = edge_features(child, "Therapieempfehlung", config)
    assert with_colon["parent_ends_in_colon"] and not without_colon["parent_ends_in_colon"]
    assert score_features(with_colon, config.feature_weights) > \
        score_features(without_colon, config.feature_weights)


def test_leading_connective_is_detected_in_both_languages():
    config = SyntacticConfig()
    assert edge_features("Jedoch ist die Datenlage begrenzt.", "x", config)["leading_connective"]
    assert edge_features("However, the evidence is limited.", "x", config)["leading_connective"]
    assert not edge_features("Die Datenlage ist begrenzt.", "x", config)["leading_connective"]


def test_syntactic_need_varies_by_parent():
    needs = compute_need(
        ctx_of([("a", "c"), ("b", "c")],
               {"a": "Zielwerte:", "b": "Zielwerte", "c": "senken auf 7%"}),
        NeedConfig(estimator="syntactic"),
    )
    assert needs[("a", "c")] > needs[("b", "c")]


# --- need: conditional surprisal ------------------------------------------

def scripted_scorer(conditioned: dict[tuple[str, str], float], baseline: float = 5.0):
    """An NLL scorer with hand-set numbers, so no model has to load."""
    def score(text: str, context: str | None = None) -> float:
        if context is None or not context.strip():
            return baseline
        return conditioned.get((text, context), baseline)
    return score


def test_delta_nll_subtracts_the_counterfactual_parent():
    texts = {"real": "REAL", "other": "OTHER", "child": "CHILD", "sib": "SIB"}
    edges = [("real", "child"), ("other", "sib")]
    ctx = ctx_of(edges, texts)
    # the real parent helps a lot; the control parent helps just as much, which
    # is the signature of shared vocabulary rather than supplied context
    scorer = scripted_scorer({("CHILD", "REAL"): 1.0, ("CHILD", "OTHER"): 1.0})
    corrected = delta_nll(ctx, SurprisalConfig(), scorer=scorer)
    assert corrected[("real", "child")] == pytest.approx(0.0)

    # the real parent helps and the control does not -> a genuine dependency
    genuine = scripted_scorer({("CHILD", "REAL"): 1.0, ("CHILD", "OTHER"): 5.0})
    assert delta_nll(ctx, SurprisalConfig(), scorer=genuine)[("real", "child")] == \
        pytest.approx(4.0)


def test_counterfactual_can_be_switched_off():
    texts = {"real": "REAL", "other": "OTHER", "child": "CHILD", "sib": "SIB"}
    ctx = ctx_of([("real", "child"), ("other", "sib")], texts)
    scorer = scripted_scorer({("CHILD", "REAL"): 1.0, ("CHILD", "OTHER"): 1.0})
    raw = delta_nll(ctx, SurprisalConfig(counterfactual="none"), scorer=scorer)
    # without the control, pure topical redundancy still reports a large drop
    assert raw[("real", "child")] == pytest.approx(4.0)


def test_counterfactual_parent_is_never_an_ancestor():
    ctx = ctx_of([("a", "b"), ("b", "c"), ("x", "y")],
                 {"a": "A", "b": "B", "c": "C", "x": "X", "y": "Y"})
    assert ctx.counterfactual_parent(("b", "c")) == "x"


def test_surprisal_normalization_clips_negatives_and_scales_to_one():
    deltas = {("a", "b"): 2.0, ("c", "d"): -1.0, ("e", "f"): 1.0}
    by_max = normalize(deltas, SurprisalConfig(normalize="max"))
    assert by_max[("a", "b")] == 1.0
    assert by_max[("c", "d")] == 0.0
    assert by_max[("e", "f")] == pytest.approx(0.5)
    logistic = normalize(deltas, SurprisalConfig(normalize="logistic"))
    assert 0.0 <= logistic[("e", "f")] < logistic[("a", "b")] <= 1.0
    assert logistic[("c", "d")] == 0.0


def test_surprisal_need_is_bounded():
    ctx = ctx_of([("n1", "n2"), ("n3", "n4")])
    scorer = scripted_scorer({})
    assert all(0.0 <= value <= 1.0
               for value in surprisal_need(ctx, SurprisalConfig(), scorer=scorer).values())


# --- need blending ---------------------------------------------------------

def test_blend_is_a_weighted_mean_of_its_parts():
    ctx = ctx_of([("a", "b")], {"a": "Zielwerte:", "b": "senken auf 7% bei diesen Patienten"})
    rules = compute_need(ctx, NeedConfig(estimator="reference_density"))
    syntax = compute_need(ctx, NeedConfig(estimator="syntactic"))
    blended = compute_need(ctx, NeedConfig(
        estimator="blend", blend_weights={"reference_density": 0.25, "syntactic": 0.75}))
    assert blended[("a", "b")] == pytest.approx(
        0.25 * rules[("a", "b")] + 0.75 * syntax[("a", "b")])


def test_blend_rejects_unknown_and_empty_weights():
    ctx = ctx_of([("a", "b")], {"a": "A", "b": "B"})
    with pytest.raises(ValueError, match="unknown need estimators"):
        compute_need(ctx, NeedConfig(estimator="blend", blend_weights={"nonsense": 1.0}))
    with pytest.raises(ValueError, match="at least one positive weight"):
        compute_need(ctx, NeedConfig(estimator="blend", blend_weights={"syntactic": 0.0}))


# --- the need * supply metric ---------------------------------------------

def test_need_supply_is_the_product_of_its_factors():
    ctx = ctx_of([("a", "b")], {"a": "Zielwerte:", "b": "senken auf 7%"})
    config = EdgeWeightConfig(
        metric="need_supply",
        need=NeedConfig(estimator="syntactic"),
        supply=SupplyConfig(metric="uniform"),
        need_supply=NeedSupplyConfig(composition="product"),
    )
    need = compute_need(ctx, config.need)
    assert compute_metric(ctx, config)[("a", "b")] == pytest.approx(need[("a", "b")])

    halved = config.model_copy(update={"supply": SupplyConfig(metric="structural")})
    ctx_list = ctx_of([("a", "b")], {"a": "Zielwerte:", "b": "senken auf 7%"},
                      {("a", "b"): "list_item"})
    assert compute_metric(ctx_list, halved)[("a", "b")] == pytest.approx(
        need[("a", "b")] * EdgeWeightConfig().list_item)


def test_need_supply_stays_in_unit_range():
    ctx = ctx_of()
    config = EdgeWeightConfig(
        metric="need_supply",
        need=NeedConfig(estimator="reference_density"),
        supply=SupplyConfig(metric="inverted_similarity",
                            similarity=SimilarityConfig(backend="bm25")),
    )
    assert all(0.0 <= value <= 1.0 for value in compute_metric(ctx, config).values())


# --- the evidence composition ---------------------------------------------

DE_POINTERS = ReferenceDensityConfig(language="de", use_spacy=False)


def evidence_config(**supply) -> EdgeWeightConfig:
    return EdgeWeightConfig(
        metric="need_supply",
        need=NeedConfig(estimator="reference_density", reference_density=DE_POINTERS),
        supply=SupplyConfig(**(supply or {"metric": "uniform"})),
    )


def test_evidence_is_the_default_composition():
    assert EdgeWeightConfig().need_supply.composition == "evidence"


def test_an_edge_nothing_can_be_said_about_stays_neutral():
    # a picture: no text, so no need, no pointers and nothing to judge independence by
    ctx = ctx_of([("a", "b")], {"a": "Abbildung 3 zeigt die Werte", "b": ""})
    assert compute_metric(ctx, evidence_config())[("a", "b")] == pytest.approx(0.5)


def test_absent_pointers_count_by_how_much_text_they_were_absent_from():
    short = "Metformin 500 mg"
    long = ("Metformin wird bei Typ-2-Diabetes als Mittel der ersten Wahl empfohlen, "
            "weil es gut verträglich ist, kein Hypoglykämierisiko hat und in großen "
            "Studien eine Senkung kardiovaskulärer Ereignisse gezeigt hat, sofern "
            "keine Kontraindikationen wie eine schwere Niereninsuffizienz vorliegen.")
    ctx = ctx_of([("h", "s"), ("h", "l")], {"h": "Therapie", "s": short, "l": long})
    weights = compute_metric(ctx, evidence_config())
    # both raise no pointer, but only the long one has shown it stands alone
    assert 0.4 < weights[("h", "s")] < 0.5
    assert weights[("h", "l")] < 0.3
    assert weights[("h", "l")] < weights[("h", "s")]


def test_a_table_is_not_judged_by_prose_signals():
    cells = " ".join(f"Wert{i} {i} mg" for i in range(30))
    ctx = ctx_of([("p", "t")], {"p": "Dosierung", "t": cells})
    ctx.units["t"] = Unit(node_id="t", text=cells, kind=KIND_TABLE)
    parts = evidence_components(ctx, evidence_config())[("p", "t")]
    assert parts["independence_evidence"] == 0.0
    assert parts["weight"] >= 0.5


def test_need_the_parent_delivers_drives_the_edge_above_neutral():
    ctx = ctx_of([("p", "c")], {"p": "Die Dosis von Metformin.",
                                "c": "Sie sollte reduziert werden."})
    parts = evidence_components(ctx, evidence_config())[("p", "c")]
    assert parts["need"] == pytest.approx(1.0)
    assert parts["independence_evidence"] == pytest.approx(0.0)
    assert parts["weight"] == pytest.approx(1.0)  # uniform supply delivers everything


def test_supply_is_not_capped_by_a_prior_floor():
    # zero measured supply must read as zero: the composite's prior floor would lift
    # it to the text_text prior and cap how far measured supply can move the edge
    # the parent only repeats the child, so neither measured component finds anything
    ctx = ctx_of([("p", "c")], {"p": "Metformin", "c": "Metformin wird empfohlen"},
                 {("p", "c"): "text"})
    config = evidence_config(metric="composite")
    assert compute_supply(ctx, config, config.supply, MetricDeps())[("p", "c")] >= 0.40
    assert evidence_components(ctx, config)[("p", "c")]["supply"] == pytest.approx(0.0)


def test_evidence_weights_stay_in_unit_range():
    config = EdgeWeightConfig(metric="need_supply",
                              need=NeedConfig(estimator="blend"),
                              supply=SupplyConfig(similarity=SimilarityConfig(backend="bm25")))
    assert all(0.0 <= value <= 1.0 for value in compute_metric(ctx_of(), config).values())


def test_half_length_sets_how_fast_absence_becomes_evidence():
    ctx = ctx_of([("h", "c")], {"h": "Therapie", "c": "Metformin wird empfohlen"})
    quick = evidence_config().model_copy(
        update={"need_supply": NeedSupplyConfig(independence_half_length=1.0)})
    slow = evidence_config().model_copy(
        update={"need_supply": NeedSupplyConfig(independence_half_length=100.0)})
    assert compute_metric(ctx, quick)[("h", "c")] < compute_metric(ctx, slow)[("h", "c")]


# --- dispatch and combination ---------------------------------------------

def test_every_metric_is_registered():
    assert set(available_metrics()) == {
        "structural", "uniform", "similarity", "inverted_similarity", "need_supply"}


def test_unknown_metric_names_the_available_ones():
    config = EdgeWeightConfig()
    config.metric = "nope"  # bypass validation, as a stale config file would
    with pytest.raises(ValueError, match="unknown edge weight metric"):
        compute_metric(ctx_of(), config)


def test_uniform_is_the_flat_baseline():
    assert set(compute_metric(ctx_of(), EdgeWeightConfig(metric="uniform")).values()) == {1.0}


def test_structural_weights_come_from_the_relation():
    ctx = ctx_of([("a", "b"), ("a", "c")], {"a": "A", "b": "B", "c": "C"},
                 {("a", "b"): "list_item", ("a", "c"): "section"})
    config = EdgeWeightConfig()
    weights = compute_metric(ctx, config)
    assert weights[("a", "b")] == config.list_item
    assert weights[("a", "c")] == config.section


def test_combination_modes():
    ctx = ctx_of([("a", "b")], {"a": "A", "b": "B"}, {("a", "b"): "text"})
    base = EdgeWeightConfig(metric="uniform")
    structural = base.text
    assert compute_edge_weights(ctx, base)[("a", "b")] == 1.0
    mean = base.model_copy(update={"combination": "mean"})
    assert compute_edge_weights(ctx, mean)[("a", "b")] == pytest.approx((structural + 1.0) / 2)
    multiply = base.model_copy(update={"combination": "multiply"})
    assert compute_edge_weights(ctx, multiply)[("a", "b")] == pytest.approx(structural)


def test_combination_is_a_no_op_for_the_structural_metric():
    ctx = ctx_of([("a", "b")], {"a": "A", "b": "B"}, {("a", "b"): "text"})
    config = EdgeWeightConfig(metric="structural", combination="multiply")
    assert compute_edge_weights(ctx, config)[("a", "b")] == config.text


def test_apply_edge_weights_rewrites_the_tuples():
    ctx = ctx_of([("n1", "n2"), ("n3", "n4")], TEXTS)
    config = EdgeWeightConfig(metric="uniform")
    rewritten = apply_edge_weights([("n1", "n2", 0.8), ("n3", "n4", 0.3)], ctx, config)
    assert rewritten == [("n1", "n2", 1.0), ("n3", "n4", 1.0)]


def test_deps_inject_the_encoder_instead_of_loading_a_model():
    config = EdgeWeightConfig(metric="similarity",
                              similarity=SimilarityConfig(backend="embedding",
                                                          embedding_model="unused"))
    weights = compute_metric(ctx_of(), config, MetricDeps(encode=fake_encode))
    assert weights[("n1", "n2")] == pytest.approx(1.0 / np.sqrt(2))


def test_a_custom_metric_can_be_registered():
    @register_metric("always_half")
    def always_half(ctx, config, deps):
        return {edge: 0.5 for edge in ctx.edges}

    config = EdgeWeightConfig()
    config.metric = "always_half"
    try:
        assert set(compute_metric(ctx_of(), config).values()) == {0.5}
    finally:
        from document2graph.edge_weights import _METRICS
        _METRICS.pop("always_half")


# --- defaults and the deprecated config -----------------------------------

def test_structural_is_the_default_so_behaviour_is_unchanged():
    config = EdgeWeightConfig()
    assert config.metric == "structural"
    assert config.combination == "replace"


def test_the_removed_relevancy_block_is_gone():
    """Dropped at 0.2.0. Its translation lives on as metric="inverted_similarity"."""
    assert not hasattr(EdgeWeightConfig(), "relevancy")


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        SimilarityConfig(alpha=1.5)
