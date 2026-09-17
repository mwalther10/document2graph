"""Graph operations a retrieval layer is built from.

Deliberately not a retrieval system: no index, no scoring, no embedding model.
What lives here are the operations that need the document graph and could not be
written without it, so that a harness can own the indexing and the metrics while
the graph walks stay with the package that owns the graph.

* :mod:`.merge`     -- index time: fold context into a node to make it a unit
* :mod:`.expand`    -- query time: grow a retrieved set using the graph
* :mod:`.calibrate` -- put two weight metrics on one footing before comparing them
"""

from .calibrate import Calibration, calibrate_min_weight, mean_unit_tokens
from .expand import ExpandedHit, expand
from .merge import (
    TokenCounter,
    approximate_token_count,
    huggingface_token_counter,
    merge_units,
)

__all__ = [
    "Calibration",
    "ExpandedHit",
    "TokenCounter",
    "approximate_token_count",
    "calibrate_min_weight",
    "expand",
    "huggingface_token_counter",
    "mean_unit_tokens",
    "merge_units",
]
