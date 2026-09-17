"""Put two weight metrics on the same footing before comparing them.

``MergePolicy.min_weight`` is a threshold on whatever scale the configured edge
weight metric produces. Those scales do not agree: ``structural`` is a handful of
fixed values between 0.1 and 1.0, ``need_supply`` is a product of two ratios and
sits far lower, and a BM25-derived one is normalized against the strongest edge
in its own document. Running two metrics at ``min_weight=0.5`` compares how much
each happens to merge at 0.5, which is a fact about the scales.

What makes them comparable is fixing the *outcome*: calibrate each metric's
threshold until the corpus has the same mean tokens per unit, then compare
retrieval at that matched budget. A metric that is better at choosing then shows
up as better retrieval for the same tokens, which is the only form the claim can
honestly take.

    from document2graph.retrieval import calibrate_min_weight, merge_units

    # the target is read off a baseline: what does merging everything cost?
    target = mean_unit_tokens(graphs, MergePolicy(strategy="all_ancestors"))

    calibration = calibrate_min_weight(graphs, MergePolicy(strategy="weighted"), target)
    assert calibration.within(0.02), calibration.relative_error
    units = [unit for graph in graphs for unit in merge_units(graph, calibration.policy)]
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from ..graph_store.document_graph import DocumentGraph
from ..models.MergePolicy import MergePolicy
from .merge import TokenCounter, approximate_token_count, merge_units


def mean_unit_tokens(graphs: Sequence[DocumentGraph], policy: MergePolicy,
                     count_tokens: TokenCounter | None = None) -> float:
    """Mean tokens per unit over a corpus, which is what a budget is matched on.

    Pooled over units rather than averaged over documents: the budget a retriever
    spends is charged per unit, so a 200-page report should weigh more here than
    a 10-page one.
    """
    total = units = 0
    for graph in graphs:
        for unit in merge_units(graph, policy, count_tokens):
            total += unit.token_count
            units += 1
    return total / units if units else 0.0


class Calibration(BaseModel):
    """The outcome of a calibration, and whether it actually worked.

    The achieved mean is returned beside the policy rather than left to be looked
    up, because the search can miss and missing silently is the failure that
    matters: two conditions reported as budget-matched when one merges half again
    as much as the other say nothing about the metrics.
    """

    policy: MergePolicy
    target_mean_tokens: float
    achieved_mean_tokens: float
    # thresholds evaluated; the reachable means are a step function of these
    candidates_evaluated: int
    exhaustive: bool

    @property
    def relative_error(self) -> float:
        if self.target_mean_tokens == 0:
            return 0.0
        return (self.achieved_mean_tokens - self.target_mean_tokens) / self.target_mean_tokens

    def within(self, tolerance: float = 0.02) -> bool:
        return abs(self.relative_error) <= tolerance


def _distinct_weights(graphs: Sequence[DocumentGraph]) -> list[float]:
    """Every edge weight the corpus contains, ascending."""
    return sorted({weight for graph in graphs for _, _, weight in graph.edges})


def calibrate_min_weight(graphs: Sequence[DocumentGraph], policy: MergePolicy,
                         target_mean_tokens: float,
                         count_tokens: TokenCounter | None = None,
                         max_iterations: int = 24,
                         exhaustive_below: int = 64) -> Calibration:
    """Set ``min_weight`` so the corpus comes out at the target mean unit size.

    **The reachable sizes are a step function**, and coarsely so for a metric with
    few distinct values: the shipped structural weights take five values, so a
    corpus merged under them can only land on a handful of means and a target
    between two of them is simply not reachable. That is a fact about the metric,
    not a failure of the search, and it is why the result carries the mean it
    achieved. Read :attr:`Calibration.relative_error` before treating two
    conditions as matched.

    Where the corpus has few distinct weights every one of them is tried, which
    is exact. Above ``exhaustive_below`` the threshold is bisected instead, which
    is sound because raising it can only merge fewer ancestors, so the mean falls
    monotonically.
    """
    count = count_tokens or approximate_token_count
    if target_mean_tokens <= 0:
        raise ValueError("target_mean_tokens must be positive")

    def mean_at(threshold: float) -> float:
        return mean_unit_tokens(graphs, policy.model_copy(update={"min_weight": threshold}),
                                count)

    weights = _distinct_weights(graphs)
    best_threshold = 0.0
    best_mean = mean_at(0.0)
    best_error = abs(best_mean - target_mean_tokens)
    evaluated = 1

    def consider(threshold: float) -> float:
        nonlocal best_threshold, best_mean, best_error, evaluated
        mean = mean_at(threshold)
        evaluated += 1
        error = abs(mean - target_mean_tokens)
        if error < best_error:
            best_threshold, best_mean, best_error = threshold, mean, error
        return mean

    exhaustive = len(weights) <= exhaustive_below
    if exhaustive:
        # a threshold exactly at a weight still admits it (the gate is >=), so the
        # steps are the weights themselves plus one just above the largest
        for threshold in weights:
            consider(threshold)
        consider(min(1.0, weights[-1] + 1e-9) if weights else 1.0)
    else:
        low, high = 0.0, 1.0
        for _ in range(max_iterations):
            middle = (low + high) / 2
            if consider(middle) > target_mean_tokens:
                low = middle  # merging too much: a higher threshold merges less
            else:
                high = middle

    return Calibration(
        policy=policy.model_copy(update={"min_weight": best_threshold}),
        target_mean_tokens=target_mean_tokens,
        achieved_mean_tokens=best_mean,
        candidates_evaluated=evaluated,
        exhaustive=exhaustive,
    )
