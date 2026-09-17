"""Parent/child content similarity, and its inverse.

Both polarities are defensible and the pipeline should be able to try either,
which is the point of keeping them apart from the score itself:

* **similarity** -- a parent about the same subject is applicable context
* **inverted similarity** -- a parent that restates the child adds nothing the
  child does not already carry; the parent worth attaching is the one that
  supplies what is missing

``inverted_similarity`` is what the pipeline computed before this refactor,
under the name "relevancy".
"""

from __future__ import annotations

from ..models.EdgeWeightConfig import SimilarityConfig
from .context import Edge, EdgeContext
from .lexical import bm25_similarity
from .semantic import Encoder, embedding_similarity


def content_similarity(ctx: EdgeContext, config: SimilarityConfig,
                       encode: Encoder | None = None) -> dict[Edge, float]:
    """Similarity per edge in [0, 1], from the configured backend."""
    if config.backend == "bm25":
        return bm25_similarity(ctx, k1=config.bm25_k1, b=config.bm25_b,
                               scoring=config.bm25_scoring)
    if config.backend == "embedding":
        return embedding_similarity(ctx, config.embedding_model, encode=encode)
    lexical = bm25_similarity(ctx, k1=config.bm25_k1, b=config.bm25_b,
                              scoring=config.bm25_scoring)
    semantic = embedding_similarity(ctx, config.embedding_model, encode=encode)
    return {edge: config.alpha * semantic[edge] + (1 - config.alpha) * lexical[edge]
            for edge in lexical}


def inverted_similarity(ctx: EdgeContext, config: SimilarityConfig,
                        encode: Encoder | None = None) -> dict[Edge, float]:
    return {edge: 1.0 - value for edge, value in content_similarity(ctx, config, encode).items()}
