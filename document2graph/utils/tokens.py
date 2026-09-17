"""Token counting shared by graph construction and retrieval.

Lives here rather than in :mod:`document2graph.retrieval` because the extractor
needs the same estimate -- ``absorb_figure_text`` decides what is a label and what
is figure prose by it -- and the extractor must not depend on the retrieval layer.
"""

from __future__ import annotations

from collections.abc import Callable

TokenCounter = Callable[[str], int]


def approximate_token_count(text: str) -> int:
    """A tokenizer-free estimate: whichever of words and quarter-characters is larger.

    Only a stand-in. A budget comparison has to charge every condition with the
    *same* tokenizer as the retriever that will embed the text, so a real
    comparison passes ``count_tokens=huggingface_token_counter(...)``; this is
    what makes the module usable, and its tests fast, without a model download.

    Two estimates because either alone is wrong in a direction that matters here:
    characters over four underestimates German compounds, word count
    underestimates a table row that is mostly punctuation.
    """
    if not text:
        return 0
    return max(len(text.split()), len(text) // 4)
