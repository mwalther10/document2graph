"""Score a saved graph's edges, without the PDF and without re-parsing.

Edge weights are computed once, during graph construction, from whatever
``EdgeWeightConfig`` the extractor was given. Comparing metrics that way costs a
full extraction per metric, and an ablation that only changes how edges are
weighted would re-run a layout model over the corpus to answer a question about
arithmetic.

Everything the metrics read is already on a ``DocumentGraph``: the texts, what
kind of unit each node is, the tree shape, and -- since schema version 3 -- the
structural relation of every edge. So a metric can be re-run against a graph read
back from disk::

    from document2graph import load_graph
    from document2graph.edge_weights import recompute_weights
    from document2graph.models import EdgeWeightConfig, NeedConfig

    graph = load_graph("data/graphs/my_document_graph.json")
    rescored = recompute_weights(graph, EdgeWeightConfig(
        metric="need_supply", need=NeedConfig(estimator="syntactic")))

A graph written before schema version 3 has no relation table. The relations are
not guessed from the nodes in that case: three of them describe how an edge was
created, and inferring them would quietly turn an ``unreferenced_media`` edge
(supply prior 1.00) into an ordinary media edge (0.50). Re-extract instead.
"""

from __future__ import annotations

from ..graph_store.document_graph import DocumentGraph
from ..models.EdgeWeightConfig import EdgeWeightConfig
from .context import (
    KIND_IMAGE,
    KIND_ROOT,
    KIND_TABLE,
    KIND_TEXT,
    Edge,
    EdgeContext,
    Unit,
)
from .deps import MetricDeps

_KINDS = {"table": KIND_TABLE, "image": KIND_IMAGE, "root": KIND_ROOT}


def edge_context_from_graph(graph: DocumentGraph,
                            include_reference_edges: bool = True) -> EdgeContext:
    """The saved graph as an edge weight metric sees it.

    Reference edges are included by default because graph construction weights
    both kinds in one pass, so that a metric normalizing against the strongest
    edge in the document normalizes against the same set here as it did there.
    """
    units = {
        snippet.snippet_id: Unit(
            node_id=snippet.snippet_id,
            text=snippet.text,
            label=snippet.label,
            level_label=snippet.level_label,
            level=snippet.level,
            sequence_no=snippet.sequence_no,
            is_grouped=snippet.is_grouped,
            kind=_KINDS.get(snippet.snippet_type, KIND_TEXT),
        )
        for snippet in graph.snippets
    }

    edges: list[Edge] = [(parent_id, child_id) for parent_id, child_id, _ in graph.edges]
    if include_reference_edges:
        edges += [(parent_id, child_id) for parent_id, child_id, _ in graph.reference_edges]

    relations = {}
    for edge in edges:
        relation = graph.relation(*edge)
        if relation is not None:
            relations[edge] = relation
    return EdgeContext(units=units, edges=edges, relations=relations)


def recompute_weights(graph: DocumentGraph, config: EdgeWeightConfig,
                      deps: MetricDeps | None = None) -> DocumentGraph:
    """A copy of ``graph`` with every edge reweighted under ``config``.

    Raises when the graph predates the relation table rather than scoring it on
    guessed relations, which would be wrong in a direction that is invisible in
    the output.
    """
    from . import compute_edge_weights  # local: the package imports this module

    if graph.edges and not graph.edge_relations:
        raise ValueError(
            f"{graph.filename}: graph was written under schema version "
            f"{graph.schema_version} and carries no edge relations, which the "
            "supply measures need; re-extract it to reweight it"
        )

    ctx = edge_context_from_graph(graph)
    weights = compute_edge_weights(ctx, config, deps)
    return graph.model_copy(update={
        "edges": [(parent_id, child_id, weights.get((parent_id, child_id), weight))
                  for parent_id, child_id, weight in graph.edges],
        "reference_edges": [(parent_id, child_id, weights.get((parent_id, child_id), weight))
                            for parent_id, child_id, weight in graph.reference_edges],
    })
