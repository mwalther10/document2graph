"""How much context this particular parent delivers to this child, in [0, 1].

``need`` asks whether a unit can be read alone; ``supply`` is the half that
tells one candidate parent from another. Three measures answer it, and the
default composes all three:

* :mod:`.antecedent`      -- of the pointers the child leaves dangling, what
                             share this parent answers. Directly interpretable,
                             but undefined on any unit that raises no pointer.
* :mod:`.complementarity` -- how much high-information content the parent adds
                             that the child does not already carry. Dense where
                             coverage is sparse, and blind to what coverage sees.
* :func:`structural_prior_weights` -- what this relation type delivers on
                             average, before looking at the instance at all.

**Composition.** The larger of the first two, floored at the third::

    supply = max(max(coverage, complementarity), prior[relation])

A maximum rather than a mean because the two measures detect disjoint phenomena
and usually only one of them fires; averaging would halve the score exactly when
the other had nothing to say. A floor rather than a fourth term in a product
because the prior is not evidence about this edge -- it is what to fall back on
when there is no evidence about this edge, which on real documents is a large
share of them: a table cell is three words with no pointers and no IDF mass, and
both measured components return near zero on it by construction.

Everything stays absolutely scaled and in [0, 1], so ``need * supply`` remains
readable as expected residual uninterpretability, and the floor is commensurable
with what it floors. That rules out normalizing by the document maximum, which
is what :func:`.lexical.bm25_similarity` and the ``"max"`` mode of
:func:`.surprisal.normalize` do -- under those, a score means "relative to the
strongest edge in this document" and a fixed prior cannot be compared with it.
"""

from __future__ import annotations

from ..models.EdgeWeightConfig import EdgeWeightConfig, StructuralPriorConfig, SupplyConfig
from .antecedent import antecedent_coverage
from .complementarity import complementarity
from .context import Edge, EdgeContext
from .counterfactual import as_supply, recentre
from .deps import MetricDeps
from .relation import supply_relation
from .similarity import content_similarity, inverted_similarity
from .structural import structural_weights, uniform_weights
from .surprisal import surprisal_need


def structural_prior_weights(ctx: EdgeContext, config: StructuralPriorConfig) -> dict[Edge, float]:
    """The configured prior of each edge's relation type."""
    return {edge: config.prior(supply_relation(ctx, edge)) for edge in ctx.edges}


def measured_components(ctx: EdgeContext, config: SupplyConfig,
                        deps: MetricDeps) -> tuple[dict[Edge, float], dict[Edge, float]]:
    """Coverage and complementarity, re-centred on control parents if configured."""
    def coverage_of(context: EdgeContext) -> dict[Edge, float]:
        return antecedent_coverage(context, config.antecedent, idf=deps.idf)

    def complement_of(context: EdgeContext) -> dict[Edge, float]:
        return complementarity(context, config.complementarity, corpus_idf=deps.idf)

    if not config.counterfactual.enabled:
        return coverage_of(ctx), complement_of(ctx)
    return (as_supply(recentre(ctx, config.counterfactual, coverage_of)),
            as_supply(recentre(ctx, config.counterfactual, complement_of)))


def composite_supply(ctx: EdgeContext, config: SupplyConfig,
                     deps: MetricDeps | None = None) -> dict[Edge, float]:
    """The larger of the two measured components, floored at the structural prior."""
    deps = deps or MetricDeps()
    coverage, complement = measured_components(ctx, config, deps)
    priors = (structural_prior_weights(ctx, config.structural_prior)
              if config.floor == "prior" and config.structural_prior.enabled
              else {})

    supply: dict[Edge, float] = {}
    for edge in ctx.edges:
        a, c = coverage.get(edge, 0.0), complement.get(edge, 0.0)
        measured = max(a, c) if config.combine == "max" else (a + c) / 2
        supply[edge] = max(measured, priors.get(edge, 0.0))
    return supply


def components(ctx: EdgeContext, config: SupplyConfig,
               deps: MetricDeps | None = None) -> dict[Edge, dict[str, float]]:
    """Every component of the composite, per edge, for reporting and ablation.

    Carries ``floor_bound``: whether the prior, not the measured evidence, is
    what produced the score. A per-relation cell where that is true of nearly
    every edge is the prior restated, and reporting its mean as a measurement
    would be a mistake.
    """
    deps = deps or MetricDeps()
    coverage, complement = measured_components(ctx, config, deps)
    priors = structural_prior_weights(ctx, config.structural_prior)
    composite = composite_supply(ctx, config, deps)
    out: dict[Edge, dict[str, float]] = {}
    for edge in ctx.edges:
        a, c, p = coverage.get(edge, 0.0), complement.get(edge, 0.0), priors.get(edge, 0.0)
        measured = max(a, c) if config.combine == "max" else (a + c) / 2
        out[edge] = {
            "antecedent_coverage": a,
            "complementarity": c,
            "structural_prior": p,
            "measured": measured,
            "supply": composite.get(edge, 0.0),
            "floor_bound": float(config.floor == "prior" and p > measured),
        }
    return out


def compute_supply(ctx: EdgeContext, config: EdgeWeightConfig,
                   supply: SupplyConfig, deps: MetricDeps) -> dict[Edge, float]:
    """Supply per edge, from the configured measure."""
    if supply.metric == "uniform":
        return uniform_weights(ctx)
    if supply.metric == "structural":
        return structural_weights(ctx, config)
    if supply.metric == "antecedent_coverage":
        return antecedent_coverage(ctx, supply.antecedent, idf=deps.idf)
    if supply.metric == "complementarity":
        return complementarity(ctx, supply.complementarity, corpus_idf=deps.idf)
    if supply.metric == "composite":
        return composite_supply(ctx, supply, deps)
    if supply.metric == "similarity":
        return content_similarity(ctx, supply.similarity, encode=deps.encode)
    if supply.metric == "surprisal":
        return surprisal_need(ctx, supply.surprisal, scorer=deps.scorer)
    return inverted_similarity(ctx, supply.similarity, encode=deps.encode)
