"""Content-based relevancy weights for document graph edges.

Complements the structural weights from ``EdgeWeightConfig`` with relevancy
weights computed from the parent and child snippet texts. All relevancy
weights are normalized to [0, 1] with inverted-similarity semantics:

* 0 -> parent and child are highly similar, so the parent adds little new
  context when looking at the child (low importance)
* 1 -> parent and child share little content, so the parent is very
  relevant to consider as context for the child

Available metrics (selected via ``RelevancyWeightConfig.metric``):

* ``"bm25"``      lexical relevancy: Okapi BM25 scored against the corpus of
                  all node texts in the document tree, clipped at zero and
                  normalized by the maximum BM25 score over all edges of the
                  tree; by default the child text is the query and the parent
                  text the document, but symmetric scoring (mean or max of both
                  directions) is available via ``bm25_scoring``
* ``"embedding"`` semantic relevancy: cosine similarity of sentence embeddings
                  from a configurable sentence-transformers model, clipped to [0, 1]
* ``"blend"``     linear blend ``alpha * embedding + (1 - alpha) * bm25``
"""

import math
import re
from collections import Counter
from typing import Callable, Mapping

import numpy as np

from ..models.EdgeWeightConfig import RelevancyWeightConfig

Edge = tuple[str, str]
WeightedEdge = tuple[str, str, float]
Encoder = Callable[[list[str]], np.ndarray]

_encoder_cache: dict[str, Encoder] = {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_score(query: list[str], doc: list[str], doc_freq: Counter, n_docs: int, avgdl: float, k1: float, b: float) -> float:
    """Okapi BM25 score of ``doc`` for ``query``, clipped at zero."""
    if not doc or not query or avgdl == 0:
        return 0.0
    term_freq = Counter(doc)
    norm = k1 * (1 - b + b * len(doc) / avgdl)
    score = 0.0
    for term in set(query):
        freq = term_freq[term]
        if freq == 0:
            continue
        idf = math.log((n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
        score += idf * (freq * (k1 + 1)) / (freq + norm)
    return max(score, 0.0)  # clip negative IDF contributions to zero


def bm25_edge_weights(edges: list[Edge], texts: Mapping[str, str], k1: float = 1.5, b: float = 0.75,
                      scoring: str = "child_query") -> dict[Edge, float]:
    """Lexical relevancy weights from Okapi BM25.

    IDF and average document length come from all node texts of the document
    tree. Scores are clipped at zero, normalized by the maximum score over all
    edges, and inverted (weight = 1 - normalized score).

    ``scoring`` controls BM25's query/document asymmetry per edge:

    * ``"child_query"``    child text as query, parent text as document
    * ``"symmetric_mean"`` mean of both directions
    * ``"symmetric_max"``  max of both directions
    """
    docs = {node_id: tokenize(text) for node_id, text in texts.items()}
    n_docs = len(docs)
    avgdl = sum(len(tokens) for tokens in docs.values()) / n_docs if n_docs else 0.0
    doc_freq = Counter(term for tokens in docs.values() for term in set(tokens))

    scores: dict[Edge, float] = {}
    for parent_id, child_id in edges:
        parent = docs.get(parent_id, [])
        child = docs.get(child_id, [])
        score = _bm25_score(child, parent, doc_freq, n_docs, avgdl, k1, b)
        if scoring != "child_query":
            reverse = _bm25_score(parent, child, doc_freq, n_docs, avgdl, k1, b)
            score = max(score, reverse) if scoring == "symmetric_max" else (score + reverse) / 2
        scores[(parent_id, child_id)] = score

    max_score = max(scores.values(), default=0.0)
    if max_score <= 0:
        return {edge: 1.0 for edge in scores}
    return {edge: 1.0 - score / max_score for edge, score in scores.items()}


def _load_encoder(model_name: str) -> Encoder:
    if model_name not in _encoder_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embedding-based edge weights: "
                "pip install sentence-transformers"
            ) from e
        model = SentenceTransformer(model_name)
        _encoder_cache[model_name] = lambda batch: model.encode(batch, show_progress_bar=False) # type: ignore
    return _encoder_cache[model_name]


def embedding_edge_weights(edges: list[Edge], texts: Mapping[str, str], model_name: str, encode: Encoder | None = None) -> dict[Edge, float]:
    """Semantic relevancy weights from sentence embeddings.

    Cosine similarity between parent and child embeddings, clipped to [0, 1]
    and inverted (weight = 1 - similarity). ``encode`` overrides the
    sentence-transformers model (mainly for testing).
    """
    if not edges:
        return {}
    encode = encode or _load_encoder(model_name)
    node_ids = sorted({node_id for edge in edges for node_id in edge})
    embeddings = np.asarray(encode([texts.get(node_id, "") for node_id in node_ids]), dtype=float)
    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms == 0] = 1.0
    unit = embeddings / norms[:, None]
    index = {node_id: i for i, node_id in enumerate(node_ids)}

    weights: dict[Edge, float] = {}
    for parent_id, child_id in edges:
        similarity = float(unit[index[parent_id]] @ unit[index[child_id]])
        weights[(parent_id, child_id)] = 1.0 - min(max(similarity, 0.0), 1.0)
    return weights


def blended_edge_weights(edges: list[Edge], texts: Mapping[str, str], model_name: str, alpha: float = 0.5,
                         k1: float = 1.5, b: float = 0.75, scoring: str = "child_query",
                         encode: Encoder | None = None) -> dict[Edge, float]:
    """Linear blend of semantic and lexical weights: alpha * embedding + (1 - alpha) * bm25."""
    lexical = bm25_edge_weights(edges, texts, k1=k1, b=b, scoring=scoring)
    semantic = embedding_edge_weights(edges, texts, model_name, encode=encode)
    return {edge: alpha * semantic[edge] + (1 - alpha) * lexical[edge] for edge in lexical}


def compute_relevancy_weights(edges: list[Edge], texts: Mapping[str, str], config: RelevancyWeightConfig,
                              encode: Encoder | None = None) -> dict[Edge, float]:
    """Relevancy weight per edge according to the configured metric."""
    if config.metric == "bm25":
        return bm25_edge_weights(edges, texts, k1=config.bm25_k1, b=config.bm25_b, scoring=config.bm25_scoring)
    if config.metric == "embedding":
        return embedding_edge_weights(edges, texts, config.embedding_model, encode=encode)
    return blended_edge_weights(edges, texts, config.embedding_model, alpha=config.alpha,
                                k1=config.bm25_k1, b=config.bm25_b, scoring=config.bm25_scoring, encode=encode)


def combine_weights(structural: float, relevancy: float, combination: str) -> float:
    if combination == "multiply":
        return structural * relevancy
    return (structural + relevancy) / 2


def apply_relevancy_weights(edges: list[WeightedEdge], texts: Mapping[str, str], config: RelevancyWeightConfig,
                            encode: Encoder | None = None) -> list[WeightedEdge]:
    """Combine the structural weight of each edge with its relevancy weight.

    ``edges`` are ``(parent_id, child_id, structural_weight)`` tuples; ``texts``
    maps node ids to their text content for the whole document tree.
    """
    relevancy = compute_relevancy_weights([(parent, child) for parent, child, _ in edges], texts, config, encode=encode)
    return [(parent, child, combine_weights(weight, relevancy[(parent, child)], config.combination))
            for parent, child, weight in edges]
