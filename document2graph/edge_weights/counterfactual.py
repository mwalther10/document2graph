"""Re-centring a supply measure on what a random parent would have scored.

Both content measures answer "is this parent informative about this child",
and on real documents most of that score is explained by things that have
nothing to do with *this* pairing: how long the parent is, how long the child
is, whether the two sit in the same domain. A heading scores low because
headings are short; a body paragraph scores high because paragraphs are long.
Those effects are constant within a relation type, which is exactly why the
structural prior predicts the measured score so well and why the measures add
so little on top of it.

Re-centring removes them. Score the child against its real parent, score it
against ``k`` control parents from the same document that are *not* its
ancestors, and report the difference::

    gap = supply(true parent) - mean(supply(control parents))

What survives is what is specific to this pair. A parent that genuinely carries
the child's missing context beats a control; a parent that merely happens to be
long and on-topic does not, because the controls are long and on-topic too.

The controls have to be matched or the gap measures the matching instead of the
dependency. Two facts about the corpus force this: complementarity rises with
parent length, so an unmatched control that is systematically shorter than the
true parent manufactures a positive gap; and the score varies far more between
relation types than within them, so a control drawn from a different relation
measures "heading versus paragraph" rather than "true versus false parent".

Read the gap as a diagnostic first and a weight second. Where it is near zero
for a relation type, the underlying measure is not detecting dependency there --
it is detecting topical similarity, and the report should say so rather than
leave it to be discovered.
"""

from __future__ import annotations

from typing import Callable

from ..models.EdgeWeightConfig import CounterfactualConfig
from .context import Edge, EdgeContext
from .lexical import terms
from .relation import supply_relation

# a measure, as this module needs to call it: edges in, score per edge out
Measure = Callable[[EdgeContext], dict[Edge, float]]


def _length(ctx: EdgeContext, node_id: str) -> int:
    return len(terms(ctx.text(node_id)))


def control_parents(ctx: EdgeContext, edge: Edge,
                    config: CounterfactualConfig) -> list[str]:
    """Up to ``config.controls`` stand-in parents for this child.

    Eligible: any node that parents at least one edge, carries text, is not the
    real parent or the child, and is not one of the child's ancestors -- an
    ancestor is a parent the child genuinely depends on, so using one as a
    control would subtract the very signal being measured.

    Selection is deterministic (nearest in length, ties by node id), so a rerun
    of the same document produces the same gap.
    """
    parent_id, child_id = edge
    blocked = {parent_id, child_id, *ctx.ancestors_of(child_id)}
    target_relation = supply_relation(ctx, edge)
    target_length = _length(ctx, parent_id)

    candidates = []
    for candidate in ctx._parent_ids:
        if candidate in blocked or not ctx.text(candidate).strip():
            continue
        if config.match_relation and supply_relation(ctx, (candidate, child_id)) != target_relation:
            continue
        candidates.append(candidate)

    if config.match_length:
        candidates.sort(key=lambda node: (abs(_length(ctx, node) - target_length), node))
    else:
        candidates.sort()
    return candidates[:config.controls]


def recentre(ctx: EdgeContext, config: CounterfactualConfig,
             measure: Measure) -> dict[Edge, float]:
    """``measure`` on each real edge, minus its mean over the control edges.

    The result is a difference in [-1, 1], not a supply value; :func:`as_supply`
    maps it back. True and control edges are scored in one pass over a shared
    context so that any document-level statistic the measure builds is identical
    for both -- scoring them separately would give the two sides different
    denominators and make the subtraction meaningless.
    """
    controls = {edge: control_parents(ctx, edge, config) for edge in ctx.edges}
    control_edges = [(control, edge[1]) for edge, nodes in controls.items() for control in nodes]
    scored = measure(ctx.with_edges(list(ctx.edges) + control_edges))

    gaps: dict[Edge, float] = {}
    for edge in ctx.edges:
        nodes = controls[edge]
        if not nodes:
            # nothing to compare against: report no evidence of specificity
            gaps[edge] = 0.0
            continue
        baseline = sum(scored.get((control, edge[1]), 0.0) for control in nodes) / len(nodes)
        gaps[edge] = scored.get(edge, 0.0) - baseline
    return gaps


def as_supply(gaps: dict[Edge, float]) -> dict[Edge, float]:
    """Map a gap onto [0, 1] for use as a supply value.

    Clipped at zero rather than rescaled: a parent that does no better than a
    random non-ancestor supplies nothing *specific* to this child, and how far
    below the controls it falls is not a quantity worth ranking.
    """
    return {edge: max(gap, 0.0) for edge, gap in gaps.items()}
