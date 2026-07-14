from typing import Literal

from pydantic import BaseModel, Field


class RelevancyWeightConfig(BaseModel):
    """Optional content-based relevancy weight combined into the edge weight.

    When ``enabled``, a relevancy weight is computed from the parent and
    child snippet texts (see ``document2graph.utils.edge_weight_lib``) and
    combined with the structural weight from ``EdgeWeightConfig``. Relevancy
    weights are normalized to [0, 1] with inverted-similarity semantics:
    0 means parent and child are highly similar (the parent adds little new
    context for the child), 1 means the parent is very relevant to consider
    when looking at the child.
    """

    enabled: bool = False
    metric: Literal["bm25", "embedding", "blend"] = "blend"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # semantic share in the blend: 1.0 = fully semantic, 0.0 = fully lexical
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_scoring: Literal["child_query", "symmetric_mean", "symmetric_max"] = "child_query"  # symmetric modes score both query/document directions and take the mean or max
    combination: Literal["mean", "multiply"] = "mean"  # how structural and relevancy weight form the total weight


class EdgeWeightConfig(BaseModel):
    """Weights assigned to graph edges by structural category.

    Higher values indicate a stronger structural connection between
    parent and child node. All weights are stored on the edge as the
    ``weight`` attribute (both in the returned edge list and in the
    exported GEXF graph).
    """

    section: float = 1.0            # heading -> subheading
    text: float = 0.8               # heading/body -> body text, subtext, footnotes
    list_item: float = 0.6          # anchor text -> grouped list item (bullets)
    media: float = 0.9              # referencing text -> image/table (matched via caption)
    unreferenced_media: float = 0.3 # fallback heading -> image/table without a caption match
    reference: float = 0.7          # mentioning text -> image/table (one edge per mention, outside the hierarchy tree)
    root: float = 0.1               # synthetic document root -> otherwise disconnected node
    relevancy: RelevancyWeightConfig = Field(default_factory=RelevancyWeightConfig)
