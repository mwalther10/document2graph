"""Turn a document graph into retrievable units by merging context into nodes.

The graph is a tree of nodes that are individually too small to retrieve: a
bullet, a table cell, a paragraph whose subject is named only in the heading
above it. Merging an ancestor into a node makes it readable and costs tokens, so
the question is which ancestors earn their tokens -- which is exactly what the
edge weights on the graph are an answer to.

Every strategy is the same walk with a different stopping rule, so a comparison
between them varies one thing::

    from document2graph import load_graph
    from document2graph.models import MergePolicy
    from document2graph.retrieval import merge_units

    graph = load_graph("data/graphs/my_document_graph.json")

    bare   = merge_units(graph, MergePolicy(strategy="none"))
    merged = merge_units(graph, MergePolicy(strategy="weighted", min_weight=0.6))
    everything = merge_units(graph, MergePolicy(strategy="all_ancestors"))

The weights come off the graph, so which metric is being tested is decided when
the graph is built (``EdgeWeightConfig`` on the extractor), not here.

**One unit per node.** Every node carrying text becomes exactly one unit, and
the strategies change what that unit *contains*, not which nodes exist. That
keeps unit counts comparable and, more importantly, keeps every piece of the
document reachable: a policy that only emitted tree leaves would silently drop
the text of any paragraph that happens to anchor a list. ``subtree`` is the
exception and says so.

The synthetic document root is never a unit of its own: it is not a region of any
page, so nothing downstream could place it or score it. It remains an ancestor
like any other, so the document title still reaches the units below it, by being
merged into them and through their breadcrumb.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..graph_store.document_graph import (
    ROOT_SNIPPET_TYPE,
    DocumentGraph,
    GraphSnippet,
)
from ..models.MergedUnit import MergedEdge, MergedUnit
from ..models.MergePolicy import MergePolicy

TokenCounter = Callable[[str], int]

# Level labels the pipeline gives a node that heads a section rather than
# carrying body text. Mirrors edge_weights.context.Unit.is_heading.
HEADING_LABELS = ("Title", "Heading", "Root")


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


def huggingface_token_counter(model_name: str) -> TokenCounter:
    """Count with a Hugging Face tokenizer, loaded once.

    The tokenizer has to be the retriever's own, or a "512 token" unit is 512 of
    something else and the budget it was calibrated to means nothing.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "counting with a Hugging Face tokenizer needs transformers, which the core "
            "package does not install: pip install 'document2graph[tokenizers]'. "
            "approximate_token_count() needs nothing and is the default."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


class DocumentTree:
    """The parent/child relations of one graph, indexed for walking upward."""

    def __init__(self, graph: DocumentGraph):
        self.nodes: dict[str, GraphSnippet] = {s.snippet_id: s for s in graph.snippets}

        # A node has one parent in the hierarchy tree, but the edge list is the
        # authority rather than the assumption: when construction has left a node
        # with several, the strongest edge is the one a merge follows, and ties go
        # to the lower id so that two runs agree.
        best: dict[str, tuple[float, str]] = {}
        self.children: dict[str, list[str]] = {}
        for parent_id, child_id, weight in graph.edges:
            self.children.setdefault(parent_id, []).append(child_id)
            current = best.get(child_id)
            if current is None or (weight, parent_id) > (current[0], current[1]):
                best[child_id] = (weight, parent_id)
        self.parent: dict[str, tuple[str, float]] = {
            child_id: (parent_id, weight) for child_id, (weight, parent_id) in best.items()
        }

        for children in self.children.values():
            children.sort(key=lambda node_id: self._order(node_id))

        # Reference edges are not part of the tree: a paragraph that mentions
        # "Tab. 3" points at that table from wherever it sits, often pages away and
        # under a different heading. Nothing else in the graph connects the two, so
        # an expansion that ignores them cannot follow a citation.
        self.references: dict[str, list[str]] = {}
        self.referenced_by: dict[str, list[str]] = {}
        for source_id, target_id, _ in graph.reference_edges:
            self.references.setdefault(source_id, []).append(target_id)
            self.referenced_by.setdefault(target_id, []).append(source_id)

    def _order(self, node_id: str) -> tuple[int, str]:
        node = self.nodes.get(node_id)
        return (node.sequence_no if node else 0, node_id)

    def text_of(self, node_id: str) -> str:
        node = self.nodes.get(node_id)
        return (node.text or "").strip() if node else ""

    def is_heading(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        return bool(node and node.level_label in HEADING_LABELS)

    def ancestors(self, node_id: str) -> list[tuple[str, float]]:
        """Every ancestor of a node, nearest first, with the weight of the edge
        that reaches it. Stops at the root and at a cycle."""
        chain: list[tuple[str, float]] = []
        seen = {node_id}
        current = node_id
        while True:
            step = self.parent.get(current)
            if step is None:
                return chain
            parent_id, weight = step
            if parent_id in seen:
                return chain
            seen.add(parent_id)
            chain.append((parent_id, weight))
            current = parent_id

    def siblings(self, node_id: str) -> list[str]:
        """The other children of this node's parent, nearest in reading order first."""
        step = self.parent.get(node_id)
        if step is None:
            return []
        here = self._order(node_id)
        others = [child for child in self.children.get(step[0], []) if child != node_id]
        return sorted(others, key=lambda other: (abs(self._order(other)[0] - here[0]), other))

    def descendants(self, node_id: str) -> list[str]:
        """The node's subtree in reading order, excluding the node itself."""
        out: list[str] = []
        stack = list(reversed(self.children.get(node_id, [])))
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            out.append(current)
            stack.extend(reversed(self.children.get(current, [])))
        return sorted(out, key=self._order)

    def is_synthetic_root(self, node_id: str) -> bool:
        """Whether a node is the placeholder root rather than a real one.

        Most documents root on their own title node, which is a real block on
        page one with a box and text of its own -- across the twenty-document
        test corpus, *every* root is one. Only a document whose title could not
        be resolved gets the synthetic stand-in, which sits on no page. Testing
        for ``snippet_id == root_id`` would therefore drop the title of nearly
        every document from the units.
        """
        node = self.nodes.get(node_id)
        return bool(node and node.snippet_type == ROOT_SNIPPET_TYPE)

    def unit_nodes(self) -> list[GraphSnippet]:
        """Every node that can be a unit: has text, and is on a page."""
        return [node for node in self.nodes.values()
                if not self.is_synthetic_root(node.snippet_id) and (node.text or "").strip()]


def _breadcrumb(tree: DocumentTree, node_id: str) -> list[str]:
    """Texts of the heading ancestors of a node, outermost first."""
    trail = [tree.text_of(ancestor_id)
             for ancestor_id, _ in tree.ancestors(node_id)
             if tree.is_heading(ancestor_id) and tree.text_of(ancestor_id)]
    return list(reversed(trail))


def _gated_ancestors(tree: DocumentTree, node_id: str, policy: MergePolicy) -> list[MergedEdge]:
    """The ancestors the weight gate allows, nearest first, before the budget."""
    if policy.strategy == "none":
        return []

    selected: list[MergedEdge] = []
    cumulative = 1.0
    child_id = node_id
    for parent_id, weight in tree.ancestors(node_id):
        cumulative *= weight
        if policy.strategy == "weighted":
            # the gates stop the walk rather than skipping a link: merging a
            # grandparent while leaving out the parent between them would build a
            # unit out of text that never sat together in the document
            if weight < policy.min_weight or cumulative < policy.min_cumulative:
                break
        selected.append(MergedEdge(parent_id=parent_id, child_id=child_id,
                                   weight=weight, cumulative=cumulative))
        child_id = parent_id
    return selected


def _join(tree: DocumentTree, member_ids: Sequence[str], separator: str) -> str:
    return separator.join(
        part for part in (tree.text_of(member_id) for member_id in member_ids) if part
    )


def _fit_budget(tree: DocumentTree, node_id: str, candidates: Sequence[MergedEdge],
                policy: MergePolicy, count: TokenCounter) -> tuple[list[MergedEdge], str]:
    """Take ancestors while the *assembled* unit still fits the budget.

    Measured on the joined text rather than by summing the parts. The two differ
    -- the separator costs tokens, and a subword tokenizer does not split a
    concatenation the way it splits the pieces -- and the sum is the one that is
    not what gets reported, so budgeting by it let units past ``max_tokens``.

    A node whose own text is already over budget merges nothing: no ancestor
    could fit, and the node cannot be split here.
    """
    own_text = tree.text_of(node_id)
    if count(own_text) > policy.max_tokens:
        return [], own_text

    accepted: list[MergedEdge] = []
    text = own_text
    for edge in candidates:
        trial = accepted + [edge]
        # `trial` runs outward from the node; the text reads the other way
        trial_text = _join(tree, [e.parent_id for e in reversed(trial)] + [node_id], policy.separator)
        if count(trial_text) > policy.max_tokens:
            break
        accepted, text = trial, trial_text
    return accepted, text


def _assemble(tree: DocumentTree, node: GraphSnippet, member_ids: Sequence[str],
              separator: str, count: TokenCounter,
              merged_edges: Sequence[MergedEdge] = (), text: str | None = None) -> MergedUnit:
    """One unit from a node and the members merged into it.

    Shared with :mod:`.expand`, which builds units from a different member set --
    a parent and the children that were retrieved under it -- but needs them to
    come out as the same type, so that budget accounting and page placement do not
    have to care which stage produced a unit.
    """
    if text is None:
        text = _join(tree, member_ids, separator)
    provenance = [entry
                  for member_id in member_ids
                  for entry in (tree.nodes[member_id].provenance
                                if member_id in tree.nodes else [])]
    return MergedUnit(
        unit_id=node.snippet_id,
        document_id=node.document_id,
        text=text,
        member_ids=list(member_ids),
        breadcrumb=_breadcrumb(tree, node.snippet_id),
        token_count=count(text),
        provenance=provenance,
        page_no=node.page_no,
        node_type=node.snippet_type,
        merged_edges=list(merged_edges),
    )


def _subtree_units(tree: DocumentTree, policy: MergePolicy, count: TokenCounter) -> list[MergedUnit]:
    """One unit per heading, holding its subtree; plus anything no heading covers.

    The text of a nested section appears in its own unit and in every heading
    above it, which is inherent to the strategy rather than a defect: it is what
    makes a section node many times the size of a window, and why these units are
    only interpretable against a retrieved-token budget.
    """
    unit_nodes = tree.unit_nodes()
    # Only a heading that becomes a unit can carry anything. The synthetic root
    # does not, so a node stranded directly on it is covered by nothing and has to
    # stand alone -- 275 nodes across the test corpus sit directly under a root.
    covering = {node.snippet_id for node in unit_nodes if tree.is_heading(node.snippet_id)}

    units: list[MergedUnit] = []
    for node in unit_nodes:
        if tree.is_heading(node.snippet_id):
            members = [node.snippet_id]
            text = tree.text_of(node.snippet_id)
            # measured on the assembled text, for the reason given in _fit_budget
            for descendant_id in tree.descendants(node.snippet_id):
                if not tree.text_of(descendant_id):
                    continue
                trial = members + [descendant_id]
                trial_text = _join(tree, trial, policy.separator)
                if count(trial_text) > policy.max_tokens:
                    break
                members, text = trial, trial_text
            units.append(_assemble(tree, node, members, policy.separator, count, [], text))
        elif not any(ancestor_id in covering
                     for ancestor_id, _ in tree.ancestors(node.snippet_id)):
            # no heading unit covers it, so nothing else would carry its text
            units.append(_assemble(tree, node, [node.snippet_id], policy.separator, count, []))
    return units


def merge_units(graph: DocumentGraph, policy: MergePolicy | None = None,
                count_tokens: TokenCounter | None = None) -> list[MergedUnit]:
    """The retrievable units of one document under ``policy``, in reading order."""
    policy = policy or MergePolicy()
    count = count_tokens or approximate_token_count
    tree = DocumentTree(graph)

    if policy.strategy == "subtree":
        units = _subtree_units(tree, policy, count)
    else:
        units = []
        for node in tree.unit_nodes():
            candidates = _gated_ancestors(tree, node.snippet_id, policy)
            edges, text = _fit_budget(tree, node.snippet_id, candidates, policy, count)
            # `edges` runs outward from the node; the text reads the other way
            members = [edge.parent_id for edge in reversed(edges)] + [node.snippet_id]
            units.append(_assemble(tree, node, members, policy.separator, count, edges, text))

    return sorted(units, key=lambda unit: tree._order(unit.unit_id))
