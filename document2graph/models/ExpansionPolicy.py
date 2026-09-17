"""How a set of retrieved nodes is grown before the answer is written.

Merging happens at index time and decides what gets embedded. Expansion happens at
query time and decides what gets *returned*, which is a different trade: index-time
context is paid for in every unit whether a query needed it or not, while expansion
is paid for only on the handful of units a query actually retrieved.

The two compose rather than compete -- embed small and precise, return complete --
and the modes here are the query-time half:

* ``none``             return the hits unchanged
* ``parent``           add each hit's parent
* ``ancestors``        add the whole chain above each hit
* ``siblings``         add the hit's neighbours under the same parent
* ``subtree``          add what hangs below the hit, for when a heading is retrieved
* ``auto_merge``       where several retrieved nodes share a parent, return the
                       parent once instead of the children separately
* ``reference_edges``  follow a mention to the figure or table it points at

``auto_merge`` is the one that needs the graph's weights rather than just its shape:
a group of siblings is only worth collapsing into their parent if that parent
actually holds them together, which is what ``min_edge_weight`` asks.
"""

from typing import Literal

from pydantic import BaseModel, Field

ExpansionMode = Literal[
    "none", "parent", "ancestors", "siblings", "subtree", "auto_merge", "reference_edges",
]


class ExpansionPolicy(BaseModel):
    """What to add to a retrieved set, and how much of it to pay for."""

    mode: ExpansionMode = "auto_merge"

    # auto_merge: how many of a parent's children must have been retrieved before
    # the parent replaces them. At 1 every hit is swallowed by its parent, which is
    # index-time merging done late and badly; 2 is the point where the retriever has
    # actually said the section is relevant rather than one line of it.
    min_siblings: int = Field(default=2, ge=1)
    # auto_merge: the parent only collapses its children when it holds them this
    # firmly. 0.0 makes the decision purely about counting hits.
    min_edge_weight: float = Field(default=0.0, ge=0.0, le=1.0)

    # ceiling on what expansion adds across the whole result, since that is what a
    # generator's context is charged. Hits are grown in rank order, so a tight
    # budget grows the best-ranked ones and leaves the tail bare.
    max_added_tokens: int = Field(default=2000, ge=0)
    # no single returned unit grows past this
    max_tokens_per_unit: int = Field(default=2048, gt=0)

    # drop a hit that has been absorbed into another returned unit. Without it a
    # parent and its children are both returned, and the shared text is charged to
    # the budget twice -- which silently inflates what expansion appears to cost.
    dedupe: bool = True

    separator: str = "\n\n"
