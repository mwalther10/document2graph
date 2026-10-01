"""Grow a retrieved set using the graph, after the index has done its work.

A retriever ranks units; this decides what each of them should come back *with*.
It is a pure graph operation over ``(node_id, score)`` pairs -- no index, no
embedding model, no query -- so a harness keeps its own retrieval and asks only
the question the graph can answer::

    from document2graph import load_graph
    from document2graph.models import ExpansionPolicy
    from document2graph.retrieval import expand

    hits = [(node_id, score), ...]          # whatever the retriever ranked
    results = expand(load_graph(path), hits, ExpansionPolicy(mode="auto_merge"))

    for result in results:
        result.unit.text        # what to put in the context
        result.unit.token_count
        result.reason           # why this unit looks the way it does
        result.absorbed_ids     # hits folded into it, which are no longer returned

The result is a list of :class:`MergedUnit`, the same type index-time merging
produces, so budget accounting and page placement do not have to know which stage
built a unit.

**Scores.** A unit built from several hits takes the best of their scores. A parent
that holds three retrieved children is at least as relevant as the best of them, and
summing would rank a section above a better-matching paragraph purely for being
bigger.

**Order.** Results come back in score order, and expansion never reorders on its
own: this is a context operation, not a re-ranker. What it does change is *which*
units exist, so a merged parent takes the rank of its best child.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from ..graph_store.document_graph import DocumentGraph
from ..models.ExpansionPolicy import ExpansionPolicy
from ..models.MergedUnit import MergedEdge, MergedUnit
from .merge import DocumentTree, TokenCounter, _assemble, _join, approximate_token_count


class ExpandedHit(BaseModel):
    """One returned unit, and the account of how it came to be."""

    unit: MergedUnit
    score: float
    # the retrieved node this unit is built around
    hit_id: str
    # "hit" when nothing was added; otherwise the mode that grew it
    reason: str = "hit"
    # nodes added by expansion, which were not themselves retrieved
    added_ids: list[str] = Field(default_factory=list)
    # other hits folded into this unit, and therefore not returned separately
    absorbed_ids: list[str] = Field(default_factory=list)
    # tokens this unit's growth charged to the expansion budget
    added_tokens: int = 0


def _candidates(tree: DocumentTree, node_id: str, policy: ExpansionPolicy) -> list[str]:
    """The nodes a mode would add to one hit, nearest or most relevant first."""
    mode = policy.mode
    if mode == "parent":
        step = tree.parent.get(node_id)
        return [step[0]] if step and step[1] >= policy.min_edge_weight else []
    if mode == "ancestors":
        return [parent_id for parent_id, weight in tree.ancestors(node_id)
                if weight >= policy.min_edge_weight]
    if mode == "siblings":
        return tree.siblings(node_id)
    if mode == "subtree":
        return tree.descendants(node_id)
    if mode == "reference_edges":
        # both directions: a retrieved paragraph should bring the table it cites,
        # and a retrieved table should bring the paragraph that explains it
        return tree.references.get(node_id, []) + tree.referenced_by.get(node_id, [])
    return []


def _ordered_members(tree: DocumentTree, node_id: str, added: Sequence[str],
                     mode: str) -> list[str]:
    """The members of a unit, in the order a reader would meet them.

    Ancestors read outermost first and the hit last, as index-time merging also
    assembles them. Everything else reads in document order, which for siblings and
    a subtree is the order they sit on the page.
    """
    if mode in ("parent", "ancestors"):
        return list(reversed(list(added))) + [node_id]
    return sorted([node_id, *added], key=lambda member: tree._order(member))


def _auto_merge_groups(tree: DocumentTree, hits: Sequence[tuple[str, float]],
                       policy: ExpansionPolicy) -> dict[str, list[str]]:
    """Parents that hold enough retrieved children to be returned in their place.

    One pass, not a walk to the root. Collapsing repeatedly would carry a section
    up to the document root as soon as two of its paragraphs matched, which is a
    statement about the tree rather than about the query.
    """
    by_parent: dict[str, list[str]] = {}
    hit_ids = {node_id for node_id, _ in hits}
    for node_id, _ in hits:
        step = tree.parent.get(node_id)
        if step is None or step[1] < policy.min_edge_weight:
            continue
        parent_id, _weight = step
        # a parent that was itself retrieved is already returned; folding its
        # children into it is handled by dedupe, not by a second unit
        if parent_id in hit_ids:
            continue
        by_parent.setdefault(parent_id, []).append(node_id)
    return {parent_id: children for parent_id, children in by_parent.items()
            if len(children) >= policy.min_siblings}


def expand(graph: DocumentGraph, hits: Sequence[tuple[str, float]],
           policy: ExpansionPolicy | None = None,
           count_tokens: TokenCounter | None = None) -> list[ExpandedHit]:
    """Grow ``hits`` into the units to return, in score order."""
    policy = policy or ExpansionPolicy()
    count = count_tokens or approximate_token_count
    tree = DocumentTree(graph)

    ranked = sorted(hits, key=lambda hit: (-hit[1], hit[0]))
    known = {node_id for node_id, _ in ranked if node_id in tree.nodes}
    ranked = [(node_id, score) for node_id, score in ranked if node_id in known]

    groups = _auto_merge_groups(tree, ranked, policy) if policy.mode == "auto_merge" else {}
    absorbed_by: dict[str, str] = {}
    for parent_id, children in groups.items():
        for child_id in children:
            absorbed_by[child_id] = parent_id

    budget = policy.max_added_tokens
    results: list[ExpandedHit] = []
    emitted_parents: set[str] = set()

    for node_id, score in ranked:
        if policy.dedupe and node_id in absorbed_by:
            parent_id = absorbed_by[node_id]
            if parent_id in emitted_parents:
                continue
            group = sorted(groups[parent_id], key=lambda child: tree._order(child))
            members = [parent_id, *group]
            best = max(s for n, s in ranked if n in groups[parent_id])
            unit, cost = _grow(tree, parent_id, members, policy, count, budget)
            if unit is None:
                # the group will not fit, so fall back to returning this hit bare
                unit, cost = _grow(tree, node_id, [node_id], policy, count, budget)
                results.append(ExpandedHit(unit=unit, score=score, hit_id=node_id))
                continue
            emitted_parents.add(parent_id)
            budget -= cost
            results.append(ExpandedHit(
                unit=unit, score=best, hit_id=parent_id, reason="auto_merge",
                added_ids=[parent_id],
                absorbed_ids=[child for child in group if child != node_id],
                added_tokens=cost,
            ))
            continue

        added = _candidates(tree, node_id, policy) if policy.mode != "none" else []
        members = _ordered_members(tree, node_id, added, policy.mode)
        unit, cost = _grow(tree, node_id, members, policy, count, budget)
        if unit is None:
            unit, cost = _grow(tree, node_id, [node_id], policy, count, budget)
            added = []
        budget -= cost
        results.append(ExpandedHit(
            unit=unit, score=score, hit_id=node_id,
            reason=policy.mode if added else "hit",
            added_ids=[member for member in members if member != node_id] if added else [],
            added_tokens=cost,
        ))

    return sorted(results, key=lambda result: (-result.score, result.hit_id))


def _grow(tree: DocumentTree, node_id: str, members: Sequence[str],
          policy: ExpansionPolicy, count: TokenCounter,
          budget: int) -> tuple[MergedUnit | None, int]:
    """Assemble a unit from ``members``, dropping the furthest additions to fit.

    Two ceilings apply and they are different questions: ``max_tokens_per_unit`` is
    how large one returned unit may be, ``budget`` is what is left of the whole
    result's expansion allowance. Returns ``(None, 0)`` when not even the first
    addition fits, so the caller can fall back to the bare hit.
    """
    node = tree.nodes[node_id]
    own_text = tree.text_of(node_id)
    own_tokens = count(own_text)
    if len(members) == 1:
        return _assemble(tree, node, [node_id], policy.separator, count), 0

    accepted = [member for member in members if member == node_id]
    additions = [member for member in members if member != node_id]
    text = own_text
    cost = 0
    for addition in additions:
        trial = _ordered_members(tree, node_id, [m for m in accepted + [addition]
                                                 if m != node_id], policy.mode)
        trial_text = _join(tree, trial, policy.separator)
        trial_tokens = count(trial_text)
        trial_cost = trial_tokens - own_tokens
        if trial_tokens > policy.max_tokens_per_unit or trial_cost > budget:
            break
        accepted, text, cost = accepted + [addition], trial_text, trial_cost

    if len(accepted) == 1:
        return None, 0
    ordered = _ordered_members(tree, node_id, [m for m in accepted if m != node_id], policy.mode)
    edges = _merged_edges(tree, node_id, ordered)
    return _assemble(tree, node, ordered, policy.separator, count, edges, text), cost


def _merged_edges(tree: DocumentTree, node_id: str, members: Sequence[str]) -> list[MergedEdge]:
    """The graph edges that justify each member, where one exists.

    Only the edges joining a member to the unit's own node or to its parent are
    recorded: siblings and referenced media are not reachable from the hit by a
    single hierarchy edge, and inventing one to fill the field would misreport what
    the expansion actually followed.
    """
    edges: list[MergedEdge] = []
    for member in members:
        if member == node_id:
            continue
        step = tree.parent.get(member)
        if step and step[0] == node_id:
            edges.append(MergedEdge(parent_id=node_id, child_id=member,
                                    weight=step[1], cumulative=step[1]))
            continue
        own = tree.parent.get(node_id)
        if own and own[0] == member:
            edges.append(MergedEdge(parent_id=member, child_id=node_id,
                                    weight=own[1], cumulative=own[1]))
    return edges
