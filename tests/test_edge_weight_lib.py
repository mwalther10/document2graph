import numpy as np
import pytest

from document2graph.models import EdgeWeightConfig, RelevancyWeightConfig
from document2graph.utils.edge_weight_lib import (
    apply_relevancy_weights,
    blended_edge_weights,
    bm25_edge_weights,
    combine_weights,
    compute_relevancy_weights,
    embedding_edge_weights,
)

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


def fake_encode(batch: list[str]) -> np.ndarray:
    # deterministic 3-dim embeddings keyed by text
    vectors = {
        TEXTS["n1"]: [1.0, 0.0, 0.0],
        TEXTS["n2"]: [1.0, 1.0, 0.0],   # cos(n1, n2) = 1/sqrt(2)
        TEXTS["n3"]: [0.0, 1.0, 0.0],
        TEXTS["n4"]: [0.0, 0.0, 1.0],   # cos(n3, n4) = 0
    }
    return np.array([vectors.get(text, [0.0, 0.0, 0.0]) for text in batch])


def test_bm25_weights_are_normalized_and_inverted():
    weights = bm25_edge_weights(EDGES, TEXTS)
    assert set(weights) == set(EDGES)
    assert all(0.0 <= w <= 1.0 for w in weights.values())
    # n1/n2 share "glucose metabolism" -> most similar edge gets weight 0 after max-normalization
    assert weights[("n1", "n2")] == 0.0
    # n3/n4 share no terms -> BM25 score clips to 0 -> maximal relevancy weight
    assert weights[("n3", "n4")] == 1.0


def test_bm25_symmetric_modes_are_direction_invariant():
    for scoring in ("symmetric_mean", "symmetric_max"):
        weights = bm25_edge_weights([("n1", "n2"), ("n2", "n1")], TEXTS, scoring=scoring)
        assert weights[("n1", "n2")] == pytest.approx(weights[("n2", "n1")])
    # child_query is asymmetric: swapping parent and child changes the raw score
    asymmetric = bm25_edge_weights([("n1", "n2"), ("n2", "n1")], TEXTS, scoring="child_query")
    assert asymmetric[("n1", "n2")] != pytest.approx(asymmetric[("n2", "n1")])


def test_bm25_symmetric_max_dominates_mean():
    # raw max >= raw mean per edge; after max-normalization and inversion the
    # most similar edge is pinned to 0 in both modes, so compare on the rest
    edges = [("n1", "n2"), ("n3", "n4"), ("n1", "n4")]
    mean_weights = bm25_edge_weights(edges, TEXTS, scoring="symmetric_mean")
    max_weights = bm25_edge_weights(edges, TEXTS, scoring="symmetric_max")
    assert all(0.0 <= w <= 1.0 for w in list(mean_weights.values()) + list(max_weights.values()))
    assert mean_weights[("n1", "n2")] == 0.0 and max_weights[("n1", "n2")] == 0.0


def test_bm25_handles_empty_and_unknown_nodes():
    weights = bm25_edge_weights([("missing", "n1"), ("n1", "n2")], TEXTS)
    assert weights[("missing", "n1")] == 1.0
    all_empty = bm25_edge_weights([("a", "b")], {"a": "", "b": ""})
    assert all_empty[("a", "b")] == 1.0


def test_embedding_weights_invert_cosine_similarity():
    weights = embedding_edge_weights(EDGES, TEXTS, model_name="unused", encode=fake_encode)
    assert weights[("n1", "n2")] == pytest.approx(1.0 - 1.0 / np.sqrt(2))
    assert weights[("n3", "n4")] == 1.0  # orthogonal embeddings


def test_blend_interpolates_between_lexical_and_semantic():
    lexical = bm25_edge_weights(EDGES, TEXTS)
    semantic = embedding_edge_weights(EDGES, TEXTS, model_name="unused", encode=fake_encode)
    for alpha in (0.0, 0.3, 1.0):
        blended = blended_edge_weights(EDGES, TEXTS, model_name="unused", alpha=alpha, encode=fake_encode)
        for edge in EDGES:
            expected = alpha * semantic[edge] + (1 - alpha) * lexical[edge]
            assert blended[edge] == pytest.approx(expected)


def test_compute_relevancy_weights_dispatches_on_metric():
    bm25_cfg = RelevancyWeightConfig(enabled=True, metric="bm25")
    symmetric_cfg = RelevancyWeightConfig(enabled=True, metric="bm25", bm25_scoring="symmetric_mean")
    emb_cfg = RelevancyWeightConfig(enabled=True, metric="embedding")
    blend_cfg = RelevancyWeightConfig(enabled=True, metric="blend", alpha=0.5)

    assert compute_relevancy_weights(EDGES, TEXTS, bm25_cfg) == bm25_edge_weights(EDGES, TEXTS)
    assert compute_relevancy_weights(EDGES, TEXTS, symmetric_cfg) == bm25_edge_weights(
        EDGES, TEXTS, scoring="symmetric_mean"
    )
    assert compute_relevancy_weights(EDGES, TEXTS, emb_cfg, encode=fake_encode) == embedding_edge_weights(
        EDGES, TEXTS, model_name="unused", encode=fake_encode
    )
    assert compute_relevancy_weights(EDGES, TEXTS, blend_cfg, encode=fake_encode) == blended_edge_weights(
        EDGES, TEXTS, model_name="unused", alpha=0.5, encode=fake_encode
    )


def test_combine_weights_modes():
    assert combine_weights(0.8, 0.5, "mean") == pytest.approx(0.65)
    assert combine_weights(0.8, 0.5, "multiply") == pytest.approx(0.4)


def test_apply_relevancy_weights_combines_with_structural_weight():
    structural_edges = [("n1", "n2", 0.8), ("n3", "n4", 1.0)]
    config = RelevancyWeightConfig(enabled=True, metric="bm25", combination="mean")
    relevancy = bm25_edge_weights(EDGES, TEXTS)

    reweighted = apply_relevancy_weights(structural_edges, TEXTS, config)

    assert reweighted[0] == ("n1", "n2", pytest.approx((0.8 + relevancy[("n1", "n2")]) / 2))
    assert reweighted[1] == ("n3", "n4", pytest.approx((1.0 + relevancy[("n3", "n4")]) / 2))


def test_relevancy_disabled_by_default():
    config = EdgeWeightConfig()
    assert config.relevancy.enabled is False
    assert 0.0 <= config.relevancy.alpha <= 1.0


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        RelevancyWeightConfig(alpha=1.5)
