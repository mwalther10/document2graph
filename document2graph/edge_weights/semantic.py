"""Semantic parent/child similarity from sentence embeddings."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .context import Edge, EdgeContext

Encoder = Callable[[list[str]], np.ndarray]

_encoder_cache: dict[str, Encoder] = {}


def load_encoder(model_name: str) -> Encoder:
    """A cached sentence-transformers encoder.

    Cached per process: a sweep over metrics loads each model once rather than
    once per document.
    """
    if model_name not in _encoder_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embedding-based edge weights: "
                'pip install "document2graph[embeddings]"'
            ) from e
        model = SentenceTransformer(model_name)
        _encoder_cache[model_name] = lambda batch: model.encode(batch, show_progress_bar=False)  # type: ignore
    return _encoder_cache[model_name]


def embedding_similarity(ctx: EdgeContext, model_name: str,
                         encode: Encoder | None = None) -> dict[Edge, float]:
    """Cosine similarity of parent and child embeddings, clipped to [0, 1].

    ``encode`` overrides the sentence-transformers model, mainly for testing.
    """
    if not ctx.edges:
        return {}
    encode = encode or load_encoder(model_name)
    node_ids = sorted({node_id for edge in ctx.edges for node_id in edge})
    embeddings = np.asarray(encode([ctx.text(node_id) for node_id in node_ids]), dtype=float)
    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms == 0] = 1.0
    unit = embeddings / norms[:, None]
    index = {node_id: i for i, node_id in enumerate(node_ids)}

    similarities: dict[Edge, float] = {}
    for parent_id, child_id in ctx.edges:
        cosine = float(unit[index[parent_id]] @ unit[index[child_id]])
        similarities[(parent_id, child_id)] = min(max(cosine, 0.0), 1.0)
    return similarities
