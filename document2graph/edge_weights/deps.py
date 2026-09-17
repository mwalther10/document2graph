"""Injectable model calls and corpus statistics, so tests never load a model.

Kept in its own module because both the metric registry and the supply metrics
need it, and importing it from the package root would make that circular.
"""

from __future__ import annotations

from dataclasses import dataclass

from .semantic import Encoder
from .surprisal import NLLScorer


@dataclass
class MetricDeps:
    """What a metric may be handed instead of loading it itself.

    ``idf`` is the corpus IDF table (see :func:`.lexical.idf_table`). The
    complementarity measure asks for it by name: its score is only comparable
    across documents when every document is weighted by the same term
    statistics, and a table built from one document's own nodes means "rare
    among the sections of this document", which is a different quantity per
    document. Without it the measure falls back to per-document IDF and says so
    in the config rather than silently changing meaning.
    """

    encode: Encoder | None = None
    scorer: NLLScorer | None = None
    idf: dict[str, float] | None = None
