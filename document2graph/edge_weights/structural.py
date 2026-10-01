"""Rule-based weights from the structural relation of an edge.

The relation is decided during graph construction (heading -> subheading,
anchor -> bullet, text -> figure, ...) and each one has a configured weight on
``EdgeWeightConfig``. This is the pipeline's original behaviour and stays the
default; it is also the rule-based arm to compare the content metrics against.
"""

from __future__ import annotations

from ..models.EdgeWeightConfig import EdgeWeightConfig
from .context import Edge, EdgeContext


def structural_weights(ctx: EdgeContext, config: EdgeWeightConfig) -> dict[Edge, float]:
    return {edge: config.structural_weight(ctx.relation(edge)) for edge in ctx.edges}


def uniform_weights(ctx: EdgeContext) -> dict[Edge, float]:
    """Every edge 1.0 -- the baseline any weighting has to beat."""
    return {edge: 1.0 for edge in ctx.edges}
