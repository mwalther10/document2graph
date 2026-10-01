"""Supply measure 1: how many of the child's dangling pointers this parent answers.

Of the pointers the child raises and cannot resolve from its own text, what
share find their antecedent in this parent. If a bullet leaves six pointers
dangling and the heading answers four, supply is 0.67.

The reading is the point. Multiply by need and the product is the share of the
child's pointers that stay unresolved even after merging -- expected residual
uninterpretability -- because ``need`` is ``unresolved / total`` and this is
``covered / unresolved``, so ``need * supply`` is ``covered / total`` exactly.
That identity holds for this measure alone, not for the composite that takes a
maximum over it.

**Acronyms are excluded by default**, which is a correction from measurement
rather than a preference. Over the 22-document German guideline corpus acronyms
are 41% of all unresolved pointer mass (2531 of 6188), and an acronym's
expansion is almost never in the ancestor: it sits in an earlier sibling
paragraph or an abbreviations appendix. Counting them puts a large, structurally
uncoverable block in the denominator and drags every score toward zero. The need
they express is real but document-scoped, not hierarchical.

**Pointers are IDF-weighted** for the same kind of reason. The most frequent
unresolved definite heads in that corpus are ``therapie``, ``patienten``,
``diabetes``, ``typ-1-diabetes`` -- domain-ubiquitous generics that are not
really anaphoric at all. Unweighted, the measure degenerates into topical
overlap on frequent nouns, which is exactly what the IDF weighting in
:mod:`.complementarity` exists to prevent; weighting both the same way keeps one
notion of informativeness across the two.
"""

from __future__ import annotations

from ..models.EdgeWeightConfig import AntecedentCoverageConfig
from . import linguistic as ling
from .context import Edge, EdgeContext
from .lexical import default_idf, terms
from .reference_density import Pointer, unresolved_pointers


def _parent_profile(text: str) -> tuple[set[str], bool]:
    """What a parent offers a pointer: its stems, and whether it names anything."""
    stems = set(terms(text))
    has_noun = any(word[:1].isupper() for word in ling.words(text)[1:]) or bool(stems)
    return stems, has_noun


def key_stems(key: str) -> list[str]:
    r"""The parent-side stems a pointer's key has to match.

    The key has to be normalized exactly as the parent is, or a hyphenated
    compound can never match. ``stem("HbA1c-Messung")`` is ``"hba1c-"`` -- the
    six-character prefix runs into the hyphen -- while the parent tokenizes on
    ``\w+`` to ``hba1c`` and ``messung``, so a parent naming the identical
    compound scored zero. Splitting the key the same way and requiring every
    part fixes that without loosening the test: "die HbA1c-Messung" resolves
    against a parent that names the HbA1c measurement, not against one that
    merely says "Messung". Bare numerals are dropped, so "Typ-1-Diabetes" asks
    for ``typ`` and ``diabet`` rather than also for ``1``.
    """
    return [stem for stem in terms(key) if not stem.isdigit()]


def resolves_in(pointer: Pointer, parent_text: str,
                profile: tuple[set[str], bool] | None = None) -> bool:
    """Whether ``parent_text`` carries the antecedent this pointer needs."""
    stems, has_noun = profile if profile is not None else _parent_profile(parent_text)
    if pointer.kind == "acronym":
        # the strictest case and the only exact one: the parent has to gloss it
        return ling.is_expanded_in(pointer.key, parent_text)
    if not pointer.key:
        # an anaphor names nothing, so the most that can be said is that the
        # parent introduces some entity for it to bind to
        return has_noun
    parts = key_stems(pointer.key)
    return bool(parts) and all(part in stems for part in parts)


def pointer_weight(pointer: Pointer, idf: dict[str, float] | None) -> float:
    """How much this pointer counts toward the ratio."""
    if idf is None:
        return 1.0
    if not pointer.key:
        return default_idf(idf)
    return idf.get(ling.stem(pointer.key), default_idf(idf))


def edge_coverage(child_text: str, parent_text: str, config: AntecedentCoverageConfig,
                  idf: dict[str, float] | None = None) -> float:
    """Antecedent coverage of one (parent, child) pair, in [0, 1]."""
    pointers = unresolved_pointers(child_text, config.pointers)
    if not pointers:
        return config.default_when_no_pointers
    profile = _parent_profile(parent_text)
    total = covered = 0.0
    for pointer in pointers:
        weight = pointer_weight(pointer, idf if config.idf_weighted else None)
        total += weight
        if resolves_in(pointer, parent_text, profile):
            covered += weight
    return covered / total if total else config.default_when_no_pointers


def measures_this_edge(ctx: EdgeContext, edge: Edge, config: AntecedentCoverageConfig) -> bool:
    """Whether coverage has anything to say about this edge at all.

    False on a heading parent when ``prose_parents_only`` is set, and that is a
    structural fact rather than a tuning choice. A definite description in a
    paragraph points at an entity introduced in a preceding *sibling*, not in
    the ancestor heading, so asking a three-word heading to contain the
    antecedents of a paragraph's descriptions asks for something headings do not
    supply. Measured on the labelled set: of 27 children that raise an
    unresolved pointer, 22 get no coverage from their parent, and on heading
    parents that zero is almost always correct rather than a miss.

    A score reported where this is False is the configured default, not a
    measurement, and the composite should weight it accordingly.
    """
    if not config.prose_parents_only:
        return True
    return not ctx.unit(edge[0]).is_heading


def antecedent_coverage(ctx: EdgeContext, config: AntecedentCoverageConfig,
                        idf: dict[str, float] | None = None) -> dict[Edge, float]:
    """Coverage per edge.

    Absolutely scaled: no division by the document's largest score, unlike
    :func:`.lexical.bm25_similarity`. That is what lets a floor at the structural
    prior mean anything, and what lets the counterfactual gap be compared across
    documents at all.
    """
    profiles: dict[str, tuple[set[str], bool]] = {}
    scores: dict[Edge, float] = {}
    for parent_id, child_id in ctx.edges:
        if not measures_this_edge(ctx, (parent_id, child_id), config):
            scores[(parent_id, child_id)] = config.default_when_no_pointers
            continue
        if parent_id not in profiles:
            profiles[parent_id] = _parent_profile(ctx.text(parent_id))
        pointers = unresolved_pointers(ctx.text(child_id), config.pointers)
        if not pointers:
            scores[(parent_id, child_id)] = config.default_when_no_pointers
            continue
        parent_text = ctx.text(parent_id)
        total = covered = 0.0
        for pointer in pointers:
            weight = pointer_weight(pointer, idf if config.idf_weighted else None)
            total += weight
            if resolves_in(pointer, parent_text, profiles[parent_id]):
                covered += weight
        scores[(parent_id, child_id)] = covered / total if total else config.default_when_no_pointers
    return scores


def coverage_support(ctx: EdgeContext, config: AntecedentCoverageConfig) -> dict[Edge, int]:
    """How many unresolved pointers each edge's score rests on.

    Report this beside the score. On the German guideline corpus two thirds of
    units raise no unresolved pointer once acronyms are excluded, and another
    sixth raise exactly one -- where the ratio can only come out 0.0 or 1.0. A
    per-relation mean built on a support of 0.4 pointers per edge is the
    configured default in disguise, and should be visible as such.
    """
    return {edge: (len(unresolved_pointers(ctx.text(edge[1]), config.pointers))
                   if measures_this_edge(ctx, edge, config) else 0)
            for edge in ctx.edges}
