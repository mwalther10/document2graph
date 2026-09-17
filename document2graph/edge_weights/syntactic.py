"""Need estimator: grammatical evidence that a unit was not written to stand alone.

Ignore meaning entirely and look at the grammar. A unit that is not a
well-formed standalone sentence was written to be read as part of something
else. All the signals are shallow:

* **no finite verb** -- "Metformin 500 mg zweimal täglich" is not a sentence,
  it is an entry in something
* **fragment** -- parses as a noun phrase or subordinate clause, never a main clause
* **leading discourse connective** -- "Jedoch", "Daher", "Zudem" are explicit
  instructions to connect to the preceding text
* **lowercase start** -- unusually informative in German, where capitalization
  is otherwise rigid and only nouns capitalize mid-sentence
* **parent ends in a colon** -- not a property of the child at all, a property
  of the edge, and close to decisive when it fires

This is why it is specifically good for bullet lists, where the semantic
measures get confused: bullets are short, so embedding similarity is unreliable
and surprisal is noisy, but "no finite verb, lowercase start, parent ends in
colon" nails the case with three cheap features.

It measures *grammatical* incompleteness, which is not the same as
*interpretive* incompleteness -- "Der Blutzucker soll vor dem Essen gemessen
werden" is a perfect sentence that still means little without the heading
naming which patients it applies to. Complement to ``reference_density``,
not a replacement.
"""

from __future__ import annotations

from ..models.EdgeWeightConfig import SyntacticConfig
from . import linguistic as ling
from .context import Edge, EdgeContext

FEATURES = (
    "no_finite_verb",
    "fragment",
    "leading_connective",
    "lowercase_start",
    "parent_ends_in_colon",
)


def edge_features(child_text: str, parent_text: str, config: SyntacticConfig) -> dict[str, bool]:
    """Every syntactic feature of one (parent, child) pair."""
    language = ling.resolve_language(config.language, child_text)
    nlp = None
    if config.use_spacy:
        model = config.spacy_model_en if language == "en" else config.spacy_model_de
        nlp = ling.load_spacy(model)
    return {
        "no_finite_verb": not ling.has_finite_verb(child_text, language, nlp),
        "fragment": ling.is_fragment(child_text, language, nlp),
        "leading_connective": ling.starts_with_connective(child_text, language),
        "lowercase_start": ling.starts_lowercase(child_text),
        "parent_ends_in_colon": ling.ends_with_colon(parent_text),
    }


def score_features(features: dict[str, bool], weights: dict[str, float]) -> float:
    """Weighted share of the features that fired, in [0, 1]."""
    total = sum(abs(weights.get(name, 0.0)) for name in features)
    if total == 0:
        return 0.0
    fired = sum(abs(weights.get(name, 0.0)) for name, value in features.items() if value)
    return fired / total


def syntactic_need(ctx: EdgeContext, config: SyntacticConfig) -> dict[Edge, float]:
    """Need per edge.

    Genuinely per-edge rather than per-node: ``parent_ends_in_colon`` depends on
    which parent is being considered, so the same bullet can need one parent
    more than another.
    """
    needs: dict[Edge, float] = {}
    for edge in ctx.edges:
        parent_id, child_id = edge
        features = edge_features(ctx.text(child_id), ctx.text(parent_id), config)
        needs[edge] = score_features(features, config.feature_weights)
    return needs
