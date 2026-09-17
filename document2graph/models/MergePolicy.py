"""How a node's ancestors are folded into the unit built around it.

A graph node on its own is often not a readable unit: a bullet under a colon, a
paragraph whose subject is named only in its heading, a table cell. Merging
ancestors into it fixes that and costs tokens, and the whole question this policy
exists to ask is *which* ancestors are worth the tokens.

The strategies are ordered by how much evidence they use, and the two endpoints
are the baselines the middle has to beat:

* ``none``          -- no merging. Every node is its own unit.
* ``all_ancestors`` -- merge everything up to the root, regardless of evidence.
  Retrieval's usual answer, and the one a weighted policy has to beat *at equal
  token budget* rather than at equal recall: withholding nothing can hardly lose
  recall, so the claim has to be the same recall for fewer tokens.
* ``weighted``      -- merge an ancestor while the edge weight says it supplies
  something. What the edge weight metrics are for.
* ``subtree``       -- one unit per heading, holding its whole subtree. The odd
  one out: it changes which nodes are units at all, rather than how much context
  each unit carries.

``min_weight`` is not comparable across metrics -- each produces its own scale,
so the same threshold merges at quite different rates -- which is why a
comparison calibrates it per condition to a common mean unit size rather than
sharing a number. See ``document2graph.retrieval.calibrate_min_weight``.
"""

from typing import Literal

from pydantic import BaseModel, Field

MergeStrategy = Literal["none", "all_ancestors", "subtree", "weighted"]


class MergePolicy(BaseModel):
    """Which ancestors are folded into a unit, and how much of them fits."""

    strategy: MergeStrategy = "weighted"

    # weighted only: an ancestor is merged while its edge weight is at least this.
    # The scale is the configured EdgeWeightConfig.metric's, so this is a number to
    # calibrate rather than a number to choose.
    min_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # weighted only: the product of the weights along the path back to the node.
    # 0.0 disables it, which leaves min_weight deciding each hop on its own; raising
    # it makes a long chain of mediocre edges stop sooner than a short strong one.
    min_cumulative: float = Field(default=0.0, ge=0.0, le=1.0)

    # a merged unit stops growing here. Ancestors are taken nearest first, so what
    # a tight budget drops is the outermost context rather than the closest.
    max_tokens: int = Field(default=512, gt=0)

    separator: str = "\n\n"
    breadcrumb_separator: str = " > "
