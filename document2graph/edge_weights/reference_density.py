"""Need estimator: the share of a unit's pointers it cannot resolve by itself.

Read a chunk on its own. Every time it uses a word that points at something,
ask whether what it points at is inside the chunk. If it is not, the chunk
cannot be read alone, and the thing it points at is upstream.

Four kinds of pointer are counted:

* **anaphors** -- pronouns. "It should be reduced to 7%." What should?
* **demonstratives** -- "this dose", "the above criteria". Explicit back-pointers.
* **definite descriptions** -- "the therapy" against "a therapy". The definite
  article claims the reader already knows which one; if the entity was never
  introduced inside the unit, the introduction happened outside it.
* **first-use acronyms** -- the unit writes "HbA1c" and never expands it.

The score is ``unresolved / total`` rather than a raw count: the ratio is what
stops a long unit from looking self-sufficient just by being long.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.EdgeWeightConfig import ReferenceDensityConfig
from . import linguistic as ling
from .context import Edge, EdgeContext


@dataclass(frozen=True)
class Pointer:
    """One outward-pointing expression, and what it points at.

    ``key`` is the term whose presence would resolve it -- the noun a
    demonstrative modifies, the head of a definite description, the acronym
    itself. ``surface`` is what stands in the text, which for a demonstrative is
    the determiner ("diese") rather than the thing it points at ("Dosis"), so
    only ``key`` can be looked for anywhere else. Anaphors have no key at all:
    a pronoun names nothing, which is exactly why the supply side cannot resolve
    one lexically.
    """

    kind: str
    surface: str
    offset: int
    resolved: bool
    key: str = ""


def _nouns_before(text: str, offset: int, language: str, nlp) -> list[str]:
    """Candidate antecedents occurring before ``offset`` in the same unit."""
    prefix = text[:offset]
    if nlp is not None:
        return [token.text for token in nlp(prefix) if token.pos_ in ("NOUN", "PROPN")]
    if language == "de":
        # German capitalizes nouns; skip the very first word, which is
        # capitalized for being sentence-initial rather than for being a noun
        tokens = ling.words(prefix)
        return [token for index, token in enumerate(tokens)
                if token[:1].isupper() and index > 0]
    return [token for token in ling.words(prefix) if len(token) > 3]


def _head_after(text: str, offset: int, surface: str) -> str | None:
    """The noun a demonstrative modifies: the next word after it."""
    rest = ling.words(text[offset + len(surface):])
    return rest[0] if rest else None


def analyze_unit(text: str, config: ReferenceDensityConfig) -> list[Pointer]:
    """Every pointer in ``text``, each marked resolved or unresolved."""
    if not text.strip():
        return []
    language = ling.resolve_language(config.language, text)
    lexicon = ling.lexicons(language)
    nlp = None
    if config.use_spacy:
        model = config.spacy_model_en if language == "en" else config.spacy_model_de
        nlp = ling.load_spacy(model)

    enabled = set(config.pointer_classes)
    pointers: list[Pointer] = []
    seen_stems: set[str] = set()

    offset = 0
    tokens: list[tuple[str, int]] = []
    for word in ling.words(text):
        offset = text.find(word, offset)
        tokens.append((word, offset))
        offset += len(word)

    for word, position in tokens:
        lowered = word.lower()
        if "anaphor" in enabled and lowered in lexicon["anaphors"]:
            # a pronoun resolves inside the unit only if some noun precedes it
            resolved = bool(_nouns_before(text, position, language, nlp))
            pointers.append(Pointer("anaphor", word, position, resolved))
        if "demonstrative" in enabled and lowered in lexicon["demonstratives"]:
            head = _head_after(text, position, word)
            # "this dose" resolves only if "dose" was already mentioned
            resolved = bool(head) and ling.stem(head) in seen_stems
            pointers.append(Pointer("demonstrative", word, position, resolved, key=head or ""))
        seen_stems.add(ling.stem(word))

    if "definite" in enabled:
        introduced: set[str] = set()
        for head, position in ling.definite_descriptions(text, language):
            key = ling.stem(head)
            # resolved when the entity was already named earlier in this unit
            resolved = key in introduced or key in {
                ling.stem(token) for token, token_pos in tokens if token_pos < position
            }
            pointers.append(Pointer("definite", head, position, resolved, key=head))
            introduced.add(key)

    if "acronym" in enabled:
        for acronym, position in ling.acronyms(text):
            pointers.append(
                Pointer("acronym", acronym, position,
                        ling.is_expanded_in(acronym, text), key=acronym)
            )

    return pointers


# analyze_unit is the expensive part of every pointer-based measure, and supply
# runs it once per *edge* where need runs it once per node. Memoized on the text
# and the settings that change the answer.
_ANALYSIS_CACHE: dict[tuple[str, str], list[Pointer]] = {}
_ANALYSIS_CACHE_MAX = 4096


def analyze_unit_cached(text: str, config: ReferenceDensityConfig) -> list[Pointer]:
    """:func:`analyze_unit`, memoized per (text, config)."""
    key = (text, config.model_dump_json())
    cached = _ANALYSIS_CACHE.get(key)
    if cached is None:
        if len(_ANALYSIS_CACHE) >= _ANALYSIS_CACHE_MAX:
            _ANALYSIS_CACHE.clear()
        cached = _ANALYSIS_CACHE[key] = analyze_unit(text, config)
    return cached


def unresolved_pointers(text: str, config: ReferenceDensityConfig) -> list[Pointer]:
    """Every pointer the unit raises and cannot answer from its own text.

    These are what a parent has to supply; :mod:`.antecedent` scores how many of
    them it does.
    """
    return [pointer for pointer in analyze_unit_cached(text, config) if not pointer.resolved]


def unit_need(text: str, config: ReferenceDensityConfig) -> float:
    """Unresolved reference density of one unit, in [0, 1]."""
    pointers = analyze_unit_cached(text, config)
    if not pointers:
        return config.default_when_no_pointers
    unresolved = sum(1 for pointer in pointers if not pointer.resolved)
    return unresolved / len(pointers)


def reference_density_need(ctx: EdgeContext, config: ReferenceDensityConfig) -> dict[Edge, float]:
    """Need per edge, broadcast from the child's own unresolved density.

    A node property, so every in-edge of a child carries the same value; what
    distinguishes one parent from another is supply, not need.
    """
    cache: dict[str, float] = {}
    needs: dict[Edge, float] = {}
    for edge in ctx.edges:
        child_id = edge[1]
        if child_id not in cache:
            cache[child_id] = unit_need(ctx.text(child_id), config)
        needs[edge] = cache[child_id]
    return needs
