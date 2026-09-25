"""Configuration for how graph edges are weighted.

An edge weight is produced by one *metric* (``EdgeWeightConfig.metric``),
optionally combined with the structural weight of the edge's relation
(``EdgeWeightConfig.combination``). The metrics are:

* ``"structural"``          rule-based: the weight of the edge's structural
                            relation (heading->subheading, anchor->bullet, ...)
* ``"uniform"``             every edge 1.0; the baseline to compare against
* ``"similarity"``          parent/child content similarity in [0, 1]
* ``"inverted_similarity"`` ``1 - similarity``; the parent is worth attaching
                            when it does *not* repeat the child
* ``"need_supply"``         how much the child depends on outside context,
                            and how much this particular parent delivers of
                            it. ``NeedSupplyConfig.composition`` decides how
                            the two combine: ``"evidence"`` (default) starts
                            every edge at a neutral 0.5 and lets evidence move
                            it either way; ``"product"`` is the older
                            ``need * supply`` with a prior-floored supply

See ``document2graph.edge_weights`` for the implementations.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Structural relation between a parent and a child node. Also the field names of
# the per-relation weights on EdgeWeightConfig, so they can be looked up by name.
StructuralRelation = Literal[
    "section", "text", "list_item", "media", "unreferenced_media", "reference", "root"
]

EdgeMetricName = Literal[
    "structural", "uniform", "similarity", "inverted_similarity", "need_supply"
]

NeedEstimatorName = Literal[
    "uniform", "reference_density", "syntactic", "surprisal", "blend"
]

SupplyMetricName = Literal[
    "uniform", "structural", "similarity", "inverted_similarity", "surprisal",
    "antecedent_coverage", "complementarity", "composite",
]

# The relation vocabulary the supply measures use: the structural relations, with
# "text" and "list_item" subdivided by what kind of node parents them. See
# document2graph.edge_weights.relation.
SupplyRelation = Literal[
    "section", "heading_text", "text_text", "heading_list", "text_list",
    "media_table", "media_figure", "unreferenced_media", "reference", "root",
]

# Measured, not chosen: the mean supply a relation type delivers, over 50 edges
# hand-labelled across six German guideline documents (5 per relation), scoring
# a parent that fully supplies the child's missing context 1.0, partly 0.5, and
# not at all 0.0. See tests/.test-data/edge_labels.json.
#
# Two caveats travel with these numbers. n=5 per cell, so one edge moving between
# "fully" and "partly" shifts a prior by 0.1 -- usable as a default, not yet an
# estimate to report. And "heading_text" averages over a split cell rather than a
# typical one: three of its five children stand alone entirely (a full-sentence
# definition under a heading needs nothing), so need, not supply, is what
# separates those edges.
MEASURED_SUPPLY_PRIORS: dict[str, float] = {
    "reference": 1.00,
    "unreferenced_media": 1.00,
    "heading_list": 0.80,
    "section": 0.75,
    "text_list": 0.67,
    "media_table": 0.60,
    "root": 0.60,
    "heading_text": 0.50,
    "media_figure": 0.50,
    "text_text": 0.40,
}


class SimilarityConfig(BaseModel):
    """Content similarity between a parent and a child text, normalized to [0, 1]."""

    backend: Literal["bm25", "embedding", "blend"] = "blend"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # semantic share in the blend: 1.0 = fully semantic, 0.0 = fully lexical
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # symmetric modes score both query/document directions and take the mean or max
    bm25_scoring: Literal["child_query", "symmetric_mean", "symmetric_max"] = "child_query"


class ReferenceDensityConfig(BaseModel):
    """Unresolved reference density: the share of pointers a unit cannot resolve itself.

    Every pointer that leaves the unit -- a pronoun with no antecedent inside it,
    a demonstrative, a definite description whose entity was never introduced, an
    acronym never expanded -- is evidence that the unit was written to be read
    below something else.
    """

    language: Literal["de", "en", "auto"] = "auto"
    # which pointer classes count; dropping the noisiest one (definite) is a
    # common ablation, since generic definites ("the kidney filters blood") are
    # not anaphoric at all but still match the rule
    pointer_classes: list[Literal["anaphor", "demonstrative", "definite", "acronym"]] = Field(
        default_factory=lambda: ["anaphor", "demonstrative", "definite", "acronym"]
    )
    # a unit containing no pointers at all resolves everything it raises; the
    # ratio is undefined there, so this is the value it takes
    default_when_no_pointers: float = 0.0
    # use spaCy POS tags and noun chunks instead of the regex lexicons when the
    # model is installed; the regex path stays the fallback and needs no extra
    # dependency (pip install "document2graph[linguistic]")
    use_spacy: bool = True
    spacy_model_de: str = "de_core_news_sm"
    spacy_model_en: str = "en_core_web_sm"


class SyntacticConfig(BaseModel):
    """Syntactic self-sufficiency: grammatical evidence that a unit is not standalone.

    A weighted mean of boolean features, normalized by the total weight of the
    features that could actually be evaluated. ``parent_ends_in_colon`` is a
    property of the *edge*, not of the child: a colon is a syntactic promise
    that the following material completes the sentence.

    This measures grammatical incompleteness, which is not the same as
    interpretive incompleteness -- a well-formed sentence can still be
    meaningless without its heading. It is a complement to
    ``reference_density``, not a replacement.
    """

    feature_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "no_finite_verb": 1.0,
            "fragment": 1.0,
            "leading_connective": 1.0,
            "lowercase_start": 1.0,
            "parent_ends_in_colon": 2.0,
        }
    )
    language: Literal["de", "en", "auto"] = "auto"
    use_spacy: bool = True
    spacy_model_de: str = "de_core_news_sm"
    spacy_model_en: str = "en_core_web_sm"


class SurprisalConfig(BaseModel):
    """Conditional surprisal: how much less surprising the parent makes the child.

    ``dNLL = NLL(child) - NLL(child | parent)``, per token so length cancels.
    A large drop means the parent carried information the child needed.

    Surprisal drops for two different reasons that look identical in the number:
    the parent genuinely supplies missing context, or the parent merely repeats
    the child's vocabulary so its tokens become locally predictable. The control
    is ``counterfactual``: condition on an unrelated parent from the same
    document and subtract. Genuine dependency collapses under the swap; topical
    redundancy largely survives it, because same-domain text shares vocabulary
    either way.
    """

    model: str = "Qwen/Qwen2.5-0.5B"  # small, multilingual, ungated
    # subtract a non-ancestor parent from the same document as the
    # topical-redundancy control, or "none" to report the raw drop
    counterfactual: Literal["none", "sibling_parent"] = "sibling_parent"
    max_context_tokens: int = 384
    max_child_tokens: int = 256
    # "max" divides by the largest dNLL over all edges (the bm25 convention);
    # "logistic" squashes with 1/(1+exp(-dNLL/scale)) re-centred on zero
    normalize: Literal["max", "logistic"] = "max"
    logistic_scale: float = 1.0
    device: str | None = None  # None -> cuda/mps if available, else cpu


class NeedConfig(BaseModel):
    """How much a child depends on context from outside itself, in [0, 1]."""

    estimator: NeedEstimatorName = "reference_density"
    # weights for estimator="blend"; keys are the other estimator names
    blend_weights: dict[str, float] = Field(
        default_factory=lambda: {"reference_density": 0.5, "syntactic": 0.5}
    )
    reference_density: ReferenceDensityConfig = Field(default_factory=ReferenceDensityConfig)
    syntactic: SyntacticConfig = Field(default_factory=SyntacticConfig)
    surprisal: SurprisalConfig = Field(default_factory=SurprisalConfig)


class CounterfactualConfig(BaseModel):
    """Score against control parents and report the difference.

    Removes what a measure would have scored for any parent of this shape, so
    that what remains is specific to this pairing. Off by default because the
    result is a *difference* rather than a supply value and changes how the
    number reads; turn it on to ask whether a measure is detecting dependency at
    all, or leave it off for the absolute reading.
    """

    enabled: bool = False
    controls: int = Field(default=8, ge=1)
    # a control from another relation type measures "heading vs paragraph"
    # instead of "true vs false parent"
    match_relation: bool = True
    # complementarity rises with parent length, so unmatched controls that are
    # systematically shorter than the true parent manufacture a positive gap
    match_length: bool = True


class AntecedentCoverageConfig(BaseModel):
    """Supply measure 1: the share of the child's dangling pointers this parent answers.

    ``pointers`` controls which pointers are counted at all. Acronyms are left
    out by default: they are 41% of unresolved pointer mass on the German
    guideline corpus, but an acronym's expansion lives in an earlier sibling or
    an appendix rather than in the ancestor, so counting them puts a
    structurally uncoverable block in the denominator. Put ``"acronym"`` back to
    see what that does.
    """

    pointers: ReferenceDensityConfig = Field(
        default_factory=lambda: ReferenceDensityConfig(
            pointer_classes=["anaphor", "demonstrative", "definite"]
        )
    )
    # weight each pointer by the IDF of its head, so that "die Patienten" in a
    # diabetes corpus counts for less than "die obengenannte Dosisanpassung"
    idf_weighted: bool = True
    # a definite description in a paragraph points at an entity introduced in a
    # preceding sibling, not in the ancestor heading, so a heading parent is not
    # something this measure can say anything about; scoring one produces a
    # correct-looking zero that is really an absence of evidence
    prose_parents_only: bool = True
    # a child with nothing dangling has no ratio; this is the value it takes
    default_when_no_pointers: float = 0.0


class ComplementarityConfig(BaseModel):
    """Supply measure 2: how much high-information content the parent adds.

    ``novel / (novel + child_mass)`` over IDF-weighted terms -- the share of the
    merged unit's informative mass that the parent contributes new.
    """

    # "corpus" uses the table passed through MetricDeps.idf and is what makes two
    # documents comparable; without one it falls back to the document's own nodes
    idf_source: Literal["document", "corpus"] = "corpus"
    # cap the novel mass at the k most informative new terms. Without a cap the
    # score rises with parent length alone, which would depress every heading
    top_k_novel_terms: int | None = Field(default=8, ge=1)
    # match terms by stem: German compounds otherwise make a parent that restates
    # the child ("Blutzuckerspiegel" against "Blutzucker") look novel
    stem_terms: bool = True
    # drop novel terms below this IDF entirely
    min_idf: float = 0.0


class StructuralPriorConfig(BaseModel):
    """Supply measure 3: what a relation type delivers before looking at the instance.

    Used as a floor under the measured components, not as a term in a product.
    Where a table cell or a three-word bullet gives the content measures nothing
    to work with -- which on real documents is a large share of all edges -- the
    prior is what keeps the weight sane instead of reading "this parent supplies
    nothing".

    ``priors`` is the full table and replacing it replaces it wholesale; any
    relation absent from it falls back to ``default``. The shipped values are
    measured (see ``MEASURED_SUPPLY_PRIORS``), which is worth keeping in mind
    before hand-editing them: they disagree with the hand-set structural weights
    on ``EdgeWeightConfig`` in places, notably rating ``unreferenced_media`` and
    ``reference`` at the top where the structural weights put them near the
    bottom.
    """

    priors: dict[str, float] = Field(default_factory=lambda: dict(MEASURED_SUPPLY_PRIORS))
    default: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = True

    @model_validator(mode="after")
    def _check_range(self) -> "StructuralPriorConfig":
        out_of_range = {name: value for name, value in self.priors.items()
                        if not 0.0 <= value <= 1.0}
        if out_of_range:
            raise ValueError(
                f"structural priors must lie in [0, 1]; got {out_of_range}"
            )
        return self

    def prior(self, relation: str) -> float:
        """The configured prior of a supply relation."""
        return float(self.priors.get(relation, self.default))


class SupplyConfig(BaseModel):
    """How much context this particular parent delivers to the child, in [0, 1].

    ``need`` is a property of the child alone; ``supply`` is what distinguishes
    one candidate parent from another.

    The default is ``composite``: the larger of antecedent coverage and
    complementarity, floored at the relation's structural prior. It is the only
    option that always returns a defensible number -- coverage has no
    denominator on most units and complementarity is thin on short ones, and the
    floor is what stops those from reading as "this parent supplies nothing" --
    and it needs no model download, being lexical and rule-based throughout.

    ``inverted_similarity``, which this default replaced, is kept because it is
    what the pipeline computed before the refactor, but note that it is
    maximized by a parent having *nothing* in common with the child, which is
    backwards as an answer to "how much does this parent deliver".
    ``surprisal`` measures delivery directly and is the most faithful reading, at
    the cost of a forward pass per edge.
    """

    metric: SupplyMetricName = "composite"
    # how the two measured components combine before the floor is applied. They
    # detect disjoint phenomena and usually only one of them fires, so a mean
    # halves the score exactly when the other measure had nothing to say
    combine: Literal["max", "mean"] = "max"
    # "prior" floors the combined score at the relation's structural prior;
    # "none" reports the measured components alone, which is the ablation
    floor: Literal["prior", "none"] = "prior"
    antecedent: AntecedentCoverageConfig = Field(default_factory=AntecedentCoverageConfig)
    complementarity: ComplementarityConfig = Field(default_factory=ComplementarityConfig)
    structural_prior: StructuralPriorConfig = Field(default_factory=StructuralPriorConfig)
    counterfactual: CounterfactualConfig = Field(default_factory=CounterfactualConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    surprisal: SurprisalConfig = Field(default_factory=SurprisalConfig)


class NeedSupplyConfig(BaseModel):
    """How ``need`` and ``supply`` combine into one weight under ``metric="need_supply"``.

    ``"evidence"`` -- the default::

        w = 0.5 * (1 + merge_evidence - independence_evidence)

        merge_evidence        = need * measured_supply
        independence_evidence = (1 - need) * n / (n + independence_half_length)

    Every edge starts at 0.5, because construction put it there: that the edge
    exists is the structural fact, and it is the same fact for every relation, so
    it is one number rather than a table of per-relation priors. Evidence then
    moves it. A child that needs context this parent measurably delivers moves
    toward 1; a child that shows it reads on its own moves toward 0; a child
    nothing can be said about -- a three-word table cell, a picture -- stays at
    0.5. That three-way split is the point: the product form scored "no pointers
    found in three words" and "no pointers found in sixty" identically, at zero.

    ``n`` is the child's word count, and ``n / (n + half_length)`` is how far the
    absence of dependency signals can be trusted as evidence of independence: at
    ``n == half_length`` it counts for half. Supply is the *measured* supply --
    under ``SupplyConfig.metric="composite"`` the prior floor is not applied, so
    no per-relation prior enters the weight.

    Under this composition ``MergePolicy.min_weight`` has a reading: below 0.5 it
    merges everything except children that show they stand alone, above 0.5 only
    edges with positive evidence of need.

    ``"product"`` is ``need * supply`` with supply as configured, floor included.
    Kept for comparison against earlier runs.
    """

    composition: Literal["evidence", "product"] = "evidence"
    # words at which "no dependency signal found" counts as half the evidence of
    # independence it would be in an arbitrarily long unit
    independence_half_length: float = Field(default=20.0, gt=0.0)


class EdgeWeightConfig(BaseModel):
    """Weights assigned to graph edges.

    The per-relation floats below are the structural weights: they are the whole
    weight under ``metric="structural"`` (the default), and stay available to
    every other metric through ``combination``. All weights are stored on the
    edge as the ``weight`` attribute, both in the returned edge list and in the
    exported GEXF graph.
    """

    section: float = 1.0            # heading -> subheading
    text: float = 0.8               # heading/body -> body text, subtext, footnotes
    list_item: float = 0.6          # anchor text -> grouped list item (bullets)
    media: float = 0.9              # referencing text -> image/table (matched via caption)
    unreferenced_media: float = 0.3 # fallback heading -> image/table without a caption match
    reference: float = 0.7          # mentioning text -> image/table (one edge per mention, outside the hierarchy tree)
    root: float = 0.1               # synthetic document root -> otherwise disconnected node

    metric: EdgeMetricName = "structural"
    # how the metric's weight combines with the edge's structural weight;
    # "replace" ignores the structural weight (and is a no-op for "structural")
    combination: Literal["replace", "mean", "multiply"] = "replace"

    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    need: NeedConfig = Field(default_factory=NeedConfig)
    supply: SupplyConfig = Field(default_factory=SupplyConfig)
    need_supply: NeedSupplyConfig = Field(default_factory=NeedSupplyConfig)

    def structural_weight(self, relation: str) -> float:
        """The configured weight of a structural relation."""
        return float(getattr(self, relation))
