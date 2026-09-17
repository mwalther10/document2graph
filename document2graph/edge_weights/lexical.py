"""Lexical parent/child similarity from Okapi BM25, and the IDF table the
content-based supply measures weight terms by."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from . import linguistic as ling
from .context import Edge, EdgeContext


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_score(query: list[str], doc: list[str], doc_freq: Counter, n_docs: int,
                avgdl: float, k1: float, b: float) -> float:
    """Okapi BM25 score of ``doc`` for ``query``, clipped at zero."""
    if not doc or not query or avgdl == 0:
        return 0.0
    term_freq = Counter(doc)
    norm = k1 * (1 - b + b * len(doc) / avgdl)
    score = 0.0
    for term in set(query):
        freq = term_freq[term]
        if freq == 0:
            continue
        idf = math.log((n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
        score += idf * (freq * (k1 + 1)) / (freq + norm)
    return max(score, 0.0)  # clip negative IDF contributions to zero


def bm25_similarity(ctx: EdgeContext, k1: float = 1.5, b: float = 0.75,
                    scoring: str = "child_query") -> dict[Edge, float]:
    """Lexical similarity per edge, normalized to [0, 1].

    IDF and average document length come from all node texts of the document
    tree, so a term that is common in this document counts for little. Scores
    are clipped at zero and divided by the largest score over all edges: the
    result is relative within one document, which is what makes weights of
    different documents comparable at all.

    ``scoring`` controls BM25's query/document asymmetry per edge:

    * ``"child_query"``    child text as query, parent text as document
    * ``"symmetric_mean"`` mean of both directions
    * ``"symmetric_max"``  max of both directions
    """
    docs = {node_id: tokenize(text) for node_id, text in ctx.texts.items()}
    n_docs = len(docs)
    avgdl = sum(len(tokens) for tokens in docs.values()) / n_docs if n_docs else 0.0
    doc_freq = Counter(term for tokens in docs.values() for term in set(tokens))

    scores: dict[Edge, float] = {}
    for parent_id, child_id in ctx.edges:
        parent = docs.get(parent_id, [])
        child = docs.get(child_id, [])
        score = _bm25_score(child, parent, doc_freq, n_docs, avgdl, k1, b)
        if scoring != "child_query":
            reverse = _bm25_score(parent, child, doc_freq, n_docs, avgdl, k1, b)
            score = max(score, reverse) if scoring == "symmetric_max" else (score + reverse) / 2
        scores[(parent_id, child_id)] = score

    max_score = max(scores.values(), default=0.0)
    if max_score <= 0:
        return {edge: 0.0 for edge in scores}
    return {edge: score / max_score for edge, score in scores.items()}


def terms(text: str, stem: bool = True) -> list[str]:
    """The terms of a text, as the supply measures count them.

    Stemmed by default, and the stemming is not optional in practice for German:
    ``Blutzuckerspiegel`` in a parent and ``Blutzucker`` in its child are the same
    term for any purpose this package has, and comparing them unstemmed makes a
    parent that merely restates the child look like it introduces new content.
    Shares :func:`.linguistic.stem` with the pointer measures so that both agree
    on what "the same term" means.
    """
    tokens = tokenize(text)
    return [ling.stem(token) for token in tokens] if stem else tokens


def idf_table(texts: Iterable[str], stem: bool = True) -> dict[str, float]:
    """Smoothed inverse document frequency per term.

    ``log((N + 1) / (df + 1)) + 1``, which unlike BM25's Robertson IDF
    (:func:`_bm25_score`) never goes negative -- a term shared by most documents
    contributes little, never a penalty. That matters because these weights are
    summed into a mass ratio, where a negative contribution has no meaning.

    Build this over the whole corpus, not one document: passing it to the
    complementarity measure through :class:`.deps.MetricDeps` is what makes the
    scores of two documents comparable.
    """
    documents = [set(terms(text, stem=stem)) for text in texts]
    n_docs = len(documents)
    if not n_docs:
        return {}
    doc_freq = Counter(term for document in documents for term in document)
    return {term: math.log((n_docs + 1) / (count + 1)) + 1.0
            for term, count in doc_freq.items()}


def default_idf(table: dict[str, float]) -> float:
    """The weight of a term the table has never seen: the rarest weight in it.

    A term absent from the corpus is at least as informative as the rarest term
    present, so falling back to the maximum is the conservative reading. An empty
    table gives every term the same weight, which makes the measures unweighted
    rather than wrong.
    """
    return max(table.values(), default=1.0)
