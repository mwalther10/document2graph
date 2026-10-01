"""How much a child depends on context from outside itself.

``need`` is the first half of ``w(child, parent) = need(child) * supply(child, parent)``.
It asks a question about the child alone -- can this unit be read on its own? --
and the three estimators answer it from different evidence:

* :mod:`.reference_density` -- pointers the unit cannot resolve itself (cheap, rules)
* :mod:`.syntactic` -- grammatical incompleteness (cheap, rules)
* :mod:`.surprisal` -- how much any real ancestor reduces the child's perplexity
  (one forward pass per candidate edge, but uniform across every relation type)

The first two are complements, not alternatives: reference density catches a
well-formed sentence that points outward, syntactic catches a fragment that
points at nothing because it has no arguments at all. ``"blend"`` runs several
and takes a weighted mean.

Every estimator returns a value *per edge* even when the underlying signal is a
property of the child, so they compose uniformly. ``syntactic`` and
``surprisal`` genuinely vary by parent; ``reference_density`` broadcasts.
"""

from __future__ import annotations

from ..models.EdgeWeightConfig import NeedConfig
from .context import Edge, EdgeContext
from .reference_density import reference_density_need
from .surprisal import NLLScorer, surprisal_need
from .syntactic import syntactic_need


def compute_need(ctx: EdgeContext, config: NeedConfig,
                 scorer: NLLScorer | None = None) -> dict[Edge, float]:
    """Need per edge, in [0, 1]."""
    estimator = config.estimator
    if estimator == "uniform":
        return {edge: 1.0 for edge in ctx.edges}
    if estimator == "reference_density":
        return reference_density_need(ctx, config.reference_density)
    if estimator == "syntactic":
        return syntactic_need(ctx, config.syntactic)
    if estimator == "surprisal":
        return surprisal_need(ctx, config.surprisal, scorer=scorer)

    weights = {name: weight for name, weight in config.blend_weights.items() if weight > 0}
    if not weights:
        raise ValueError("need.blend_weights must contain at least one positive weight")
    unknown = set(weights) - {"uniform", "reference_density", "syntactic", "surprisal"}
    if unknown:
        raise ValueError(f"unknown need estimators in blend_weights: {sorted(unknown)}")

    total = sum(weights.values())
    blended = {edge: 0.0 for edge in ctx.edges}
    for name, weight in weights.items():
        part = compute_need(ctx, config.model_copy(update={"estimator": name}), scorer=scorer)
        for edge in blended:
            blended[edge] += weight * part.get(edge, 0.0) / total
    return blended
