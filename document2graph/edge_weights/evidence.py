"""The ``"evidence"`` composition of ``need`` and ``supply``: a neutral start, moved by evidence.

``need * supply`` can only trim. With need broadcast from pointer density, a unit
that raises no pointer has need 0 and its edge weighs 0 whatever the parent
delivers; with supply floored at a per-relation prior, the supply side can at most
scale a weight by 1 / floor. And a zero there conflates two different findings:
*nothing could be measured* (three words, a picture) and *the child was measured
to stand alone* (sixty words, complete sentences, nothing dangling).

This composition keeps those apart::

    w = 0.5 * (1 + merge_evidence - independence_evidence)

* **0.5** is the edge existing. Construction attached this child to this parent,
  and that is the same structural fact whatever the relation, so it is one number
  rather than a table of priors that would need measuring and tuning.
* **merge_evidence** ``= need * measured_supply`` moves the edge toward 1: the
  child needs context, and this parent measurably delivers it. No prior floor, so
  supply spans its full range instead of a 2.5x band.
* **independence_evidence** ``= (1 - need) * confidence`` moves it toward 0: the
  child shows no sign of needing anything, weighted by how much text there was to
  show it in. ``confidence = n / (n + half_length)`` over the child's words, and
  0 for a table or a figure: the need estimators read grammar and pointers, which
  a serialized table has none of, so what they report there is silence rather
  than a finding that it stands alone.

The two terms cannot both be large -- one carries ``need``, the other
``1 - need`` -- so the weight stays in [0, 1].
"""

from __future__ import annotations

from ..models.EdgeWeightConfig import EdgeWeightConfig
from . import linguistic as ling
from .context import Edge, EdgeContext, Unit
from .deps import MetricDeps
from .need import compute_need
from .supply import compute_supply

NEUTRAL = 0.5


def measured_supply(ctx: EdgeContext, config: EdgeWeightConfig,
                    deps: MetricDeps) -> dict[Edge, float]:
    """Supply per edge as the evidence measures it: the composite with no prior floor."""
    supply = config.supply
    if supply.metric == "composite":
        supply = supply.model_copy(update={"floor": "none"})
    return compute_supply(ctx, config, supply, deps)


def confidence(unit: Unit, half_length: float) -> float:
    """How far missing dependency signals in a unit count as evidence of independence."""
    if unit.is_media:
        return 0.0
    n = len(ling.words(unit.text))
    return n / (n + half_length)


def evidence_components(ctx: EdgeContext, config: EdgeWeightConfig,
                        deps: MetricDeps | None = None) -> dict[Edge, dict[str, float]]:
    """Every term of the evidence weight, per edge, for reporting and ablation."""
    deps = deps or MetricDeps()
    need = compute_need(ctx, config.need, scorer=deps.scorer)
    supply = measured_supply(ctx, config, deps)
    half_length = config.need_supply.independence_half_length

    confidences: dict[str, float] = {}
    out: dict[Edge, dict[str, float]] = {}
    for edge in ctx.edges:
        child_id = edge[1]
        if child_id not in confidences:
            confidences[child_id] = confidence(ctx.unit(child_id), half_length)
        n, s, c = need.get(edge, 0.0), supply.get(edge, 0.0), confidences[child_id]
        merge = n * s
        independence = (1.0 - n) * c
        out[edge] = {
            "need": n,
            "supply": s,
            "confidence": c,
            "merge_evidence": merge,
            "independence_evidence": independence,
            "weight": NEUTRAL * (1.0 + merge - independence),
        }
    return out


def evidence_weights(ctx: EdgeContext, config: EdgeWeightConfig,
                     deps: MetricDeps | None = None) -> dict[Edge, float]:
    """The evidence composition's weight per edge, in [0, 1]."""
    return {edge: parts["weight"]
            for edge, parts in evidence_components(ctx, config, deps).items()}
