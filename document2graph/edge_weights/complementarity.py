"""Supply measure 2: how much high-information content the parent adds.

Terms that are in the parent and not in the child, weighted by IDF, against what
the child already carries::

    novel  = sum of idf(t) for the top-k terms t in parent \\ child
    score  = novel / (novel + child_mass)

Read it as the share of the merged unit's high-information mass that the parent
contributes new. A heading that merely restates the paragraph's topic scores
near zero, correctly, because the IDF weighting gives shared function words and
domain-ubiquitous terms almost nothing.

This is the half that catches what :mod:`.antecedent` cannot: scoping content
that is semantically required but never grammatically referenced. A heading
reading "Patients over 65" supplies a qualifier the bullet below it never
pronominalizes, so no pointer measure sees anything and this one fires. The two
detect disjoint phenomena, which is why the composite takes a maximum rather
than a mean.

**Top-k is not a tuning knob, it is a correction.** Without it the score
increases monotonically with parent length: more parent text, more novel mass,
higher supply. Headings are short, so the whole ``heading_*`` row of the report
matrix would be depressed by an artifact of length rather than by supplying
less. Capping at the k most informative novel terms lets a heading's three
high-IDF qualifiers compete with a paragraph's forty low-IDF ones, which is also
the truer reading: "Patients over 65" supplies one qualifier, and the forty
extra terms of a paragraph are not forty qualifiers.

**A very short child scores high against almost any parent**, because its own
mass is the denominator. That is the stated reading working as intended rather
than a defect -- against a three-word bullet a heading really does contribute
most of the merged unit's information -- but it means a high score on a short
child is not evidence that *this* parent in particular was the right one. That
distinction is what the counterfactual gap below measures, and it is why short
units are the ones the structural prior was put there to carry.

**The measure does not punish an off-topic parent.** An unrelated heading adds
plenty of novel high-IDF mass and scores well. That cannot be fixed from inside
without smuggling similarity back in, and it is precisely what the
counterfactual gap is for: supply from the true parent minus supply from a
control parent in the same document. Where that gap is small for a relation
type, this measure is reading topical novelty rather than dependency, and the
report should say so.
"""

from __future__ import annotations

from ..models.EdgeWeightConfig import ComplementarityConfig
from .context import Edge, EdgeContext
from .lexical import default_idf, idf_table, terms


def resolve_idf(ctx: EdgeContext, config: ComplementarityConfig,
                corpus_idf: dict[str, float] | None) -> dict[str, float]:
    """The IDF table to weight with.

    ``idf_source="corpus"`` uses the injected table when there is one. There is
    no way to conjure corpus statistics from a single document, so without one
    this falls back to the document's own nodes -- the same table BM25 builds,
    meaning "rare among the sections of this document". That is a coherent
    quantity and fine within one document; it is only across documents that it
    stops being comparable, which is why the fallback is worth knowing about.
    """
    if config.idf_source == "corpus" and corpus_idf:
        return corpus_idf
    return idf_table(ctx.texts.values(), stem=config.stem_terms)


def novel_mass(parent_text: str, child_text: str, config: ComplementarityConfig,
               idf: dict[str, float]) -> tuple[float, float]:
    """``(novel, child_mass)`` for one pair, in IDF units."""
    fallback = default_idf(idf)
    child_terms = set(terms(child_text, stem=config.stem_terms))
    parent_terms = set(terms(parent_text, stem=config.stem_terms))

    def weight(term: str) -> float:
        return idf.get(term, fallback)

    novel_weights = sorted((weight(term) for term in parent_terms - child_terms), reverse=True)
    novel_weights = [w for w in novel_weights if w >= config.min_idf]
    if config.top_k_novel_terms:
        novel_weights = novel_weights[:config.top_k_novel_terms]
    child_mass = sum(weight(term) for term in child_terms)
    return sum(novel_weights), child_mass


def edge_complementarity(parent_text: str, child_text: str, config: ComplementarityConfig,
                         idf: dict[str, float]) -> float:
    """Complementarity of one (parent, child) pair, in [0, 1]."""
    novel, child_mass = novel_mass(parent_text, child_text, config, idf)
    total = novel + child_mass
    return novel / total if total > 0 else 0.0


def complementarity(ctx: EdgeContext, config: ComplementarityConfig,
                    corpus_idf: dict[str, float] | None = None) -> dict[Edge, float]:
    """Complementarity per edge, absolutely scaled to [0, 1]."""
    idf = resolve_idf(ctx, config, corpus_idf)
    return {(parent_id, child_id): edge_complementarity(
                ctx.text(parent_id), ctx.text(child_id), config, idf)
            for parent_id, child_id in ctx.edges}
