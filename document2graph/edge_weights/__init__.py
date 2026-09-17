"""Pluggable edge weight metrics for the document graph.

One metric produces the weight of every edge of a document. Which one runs is
``EdgeWeightConfig.metric``; how its result combines with the edge's structural
weight is ``EdgeWeightConfig.combination``.

    structural           the relation's configured weight (default, unchanged behaviour)
    uniform              1.0 everywhere -- the baseline
    similarity           parent/child content similarity
    inverted_similarity  1 - similarity (what "relevancy" used to compute)
    need_supply          need(child) * supply(child, parent)

``need_supply`` is the new family: how much the child depends on outside
context, times how much this particular parent delivers of it. ``need`` is a
question about the child alone (see :mod:`.need`); ``supply`` is what
distinguishes one candidate parent from another.

Registering another metric takes no changes here::

    from document2graph.edge_weights import register_metric

    @register_metric("my_metric")
    def my_metric(ctx, config, deps):
        return {edge: 1.0 for edge in ctx.edges}
"""

from __future__ import annotations

from typing import Callable

from ..models.EdgeWeightConfig import EdgeWeightConfig, SupplyConfig
from .antecedent import antecedent_coverage
from .complementarity import complementarity
from .context import (
    KIND_IMAGE,
    KIND_ROOT,
    KIND_TABLE,
    KIND_TEXT,
    RELATION_LIST_ITEM,
    RELATION_MEDIA,
    RELATION_REFERENCE,
    RELATION_ROOT,
    RELATION_SECTION,
    RELATION_TEXT,
    RELATION_UNREFERENCED_MEDIA,
    Edge,
    EdgeContext,
    Unit,
    WeightedEdge,
)
from .counterfactual import as_supply, control_parents, recentre
from .deps import MetricDeps
from .from_graph import edge_context_from_graph, recompute_weights
from .lexical import idf_table
from .need import compute_need
from .relation import SUPPLY_RELATIONS, supply_relation
from .semantic import Encoder
from .similarity import content_similarity, inverted_similarity
from .structural import structural_weights, uniform_weights
from .supply import (components, composite_supply, compute_supply, measured_components,
                     structural_prior_weights)
from .surprisal import NLLScorer, surprisal_need

MetricFn = Callable[[EdgeContext, EdgeWeightConfig, MetricDeps], dict[Edge, float]]

_METRICS: dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    def decorate(fn: MetricFn) -> MetricFn:
        _METRICS[name] = fn
        return fn
    return decorate


def available_metrics() -> list[str]:
    return sorted(_METRICS)


@register_metric("structural")
def _structural(ctx: EdgeContext, config: EdgeWeightConfig, deps: MetricDeps) -> dict[Edge, float]:
    return structural_weights(ctx, config)


@register_metric("uniform")
def _uniform(ctx: EdgeContext, config: EdgeWeightConfig, deps: MetricDeps) -> dict[Edge, float]:
    return uniform_weights(ctx)


@register_metric("similarity")
def _similarity(ctx: EdgeContext, config: EdgeWeightConfig, deps: MetricDeps) -> dict[Edge, float]:
    return content_similarity(ctx, config.similarity, encode=deps.encode)


@register_metric("inverted_similarity")
def _inverted(ctx: EdgeContext, config: EdgeWeightConfig, deps: MetricDeps) -> dict[Edge, float]:
    return inverted_similarity(ctx, config.similarity, encode=deps.encode)


@register_metric("need_supply")
def _need_supply(ctx: EdgeContext, config: EdgeWeightConfig, deps: MetricDeps) -> dict[Edge, float]:
    need = compute_need(ctx, config.need, scorer=deps.scorer)
    supply = compute_supply(ctx, config, config.supply, deps)
    return {edge: need.get(edge, 0.0) * supply.get(edge, 0.0) for edge in ctx.edges}


def compute_metric(ctx: EdgeContext, config: EdgeWeightConfig,
                   deps: MetricDeps | None = None) -> dict[Edge, float]:
    """The configured metric's weight per edge, before combination."""
    try:
        metric = _METRICS[config.metric]
    except KeyError:
        raise ValueError(
            f"unknown edge weight metric {config.metric!r}; available: {available_metrics()}"
        ) from None
    return metric(ctx, config, deps or MetricDeps())


def combine(structural: float, metric: float, combination: str) -> float:
    if combination == "replace":
        return metric
    if combination == "multiply":
        return structural * metric
    return (structural + metric) / 2


def compute_edge_weights(ctx: EdgeContext, config: EdgeWeightConfig,
                         deps: MetricDeps | None = None) -> dict[Edge, float]:
    """Final weight per edge: the metric, combined with the structural weight."""
    metric = compute_metric(ctx, config, deps)
    if config.combination == "replace" or config.metric == "structural":
        return {edge: metric.get(edge, 0.0) for edge in ctx.edges}
    return {edge: combine(config.structural_weight(ctx.relation(edge)),
                          metric.get(edge, 0.0), config.combination)
            for edge in ctx.edges}


def apply_edge_weights(edges: list[WeightedEdge], ctx: EdgeContext, config: EdgeWeightConfig,
                       deps: MetricDeps | None = None) -> list[WeightedEdge]:
    """Rewrite the weight of every ``(parent, child, weight)`` tuple."""
    weights = compute_edge_weights(ctx, config, deps)
    return [(parent, child, weights.get((parent, child), weight))
            for parent, child, weight in edges]


__all__ = [
    "SUPPLY_RELATIONS",
    "Edge",
    "EdgeContext",
    "MetricDeps",
    "Unit",
    "WeightedEdge",
    "KIND_IMAGE",
    "KIND_ROOT",
    "KIND_TABLE",
    "KIND_TEXT",
    "RELATION_LIST_ITEM",
    "RELATION_MEDIA",
    "RELATION_REFERENCE",
    "RELATION_ROOT",
    "RELATION_SECTION",
    "RELATION_TEXT",
    "RELATION_UNREFERENCED_MEDIA",
    "antecedent_coverage",
    "apply_edge_weights",
    "available_metrics",
    "complementarity",
    "components",
    "composite_supply",
    "control_parents",
    "compute_edge_weights",
    "compute_metric",
    "compute_need",
    "compute_supply",
    "content_similarity",
    "edge_context_from_graph",
    "idf_table",
    "measured_components",
    "recentre",
    "inverted_similarity",
    "recompute_weights",
    "register_metric",
    "structural_prior_weights",
    "supply_relation",
]
