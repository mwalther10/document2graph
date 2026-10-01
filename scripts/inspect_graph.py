"""Read back the saved graphs and check the hierarchy they encode.

Works off the ``*_graph.json`` written under tests/.test-data/graphs/ by the
extractor and the test runs, so it never touches docling: run the pipeline once,
then review its output as often as needed.

The three views answer three different questions:

*   the **summary** (no arguments) puts one line per document next to the others,
    which is where a document that came out unlike the rest of the corpus shows
    up -- no body text, everything on the root, a depth no layout explains.
*   the **tree** prints the hierarchy as the graph holds it. ``--headings`` cuts
    it down to the outline, which is what a PDF can be read against page by page.
*   the **checks** look for the shapes that are wrong on their face: content left
    on the root, a child ranked above its parent, a heading the length of a
    paragraph, an edge spanning half the document, a running header read as
    content.

None of it is a verdict. A check that fires is a place to look at the PDF, and
several of them fire on layouts that are simply unusual.

Usage:
    uv run python scripts/inspect_graph.py                     # summary of every document
    uv run python scripts/inspect_graph.py 054                 # summary of one
    uv run python scripts/inspect_graph.py 054 --tree          # its full tree
    uv run python scripts/inspect_graph.py 054 --tree --headings
    uv run python scripts/inspect_graph.py 054 --tree --region sidebar
    uv run python scripts/inspect_graph.py --checks            # diagnostics, whole corpus
"""

import argparse
import collections
import glob
import json
import os

GRAPH_DIR = "tests/.test-data/graphs"
# a heading this long is a paragraph docling labelled as a section header
MAX_HEADING_CHARS = 120
# a parent this far from its child is rarely the section it sits in
MAX_EDGE_PAGE_SPAN = 2
# a text repeated this often across a document is usually a running header
MIN_REPEATS = 3
MIN_REPEAT_CHARS = 25


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def build(doc: dict) -> tuple[dict, dict, dict]:
    """(nodes by id, children by parent id, parent by child id), children in reading order."""
    nodes = {snippet["snippet_id"]: snippet for snippet in doc["snippets"]}
    children: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    parent: dict[str, tuple[str, float]] = {}
    for head, tail, weight in doc["edges"]:
        children[head].append((tail, weight))
        parent[tail] = (head, weight)
    for kids in children.values():
        kids.sort(key=lambda kv: nodes.get(kv[0], {}).get("sequence_no", float("inf")))
    return nodes, children, parent


def text_of(node: dict, limit: int | None = None) -> str:
    text = " ".join((node.get("text") or "").split())
    return text[:limit] if limit else text


def depths(nodes: dict, parent: dict, root: str) -> dict[str, int]:
    """Depth of every node below the root. Cycles are cut where they close."""
    depth: dict[str, int] = {}

    def walk(node_id: str, seen: frozenset) -> int:
        if node_id in depth:
            return depth[node_id]
        above = parent.get(node_id)
        if above is None or node_id == root or node_id in seen:
            return 0
        depth[node_id] = walk(above[0], seen | {node_id}) + 1
        return depth[node_id]

    return {node_id: walk(node_id, frozenset()) for node_id in nodes if node_id != root}


def summary(path: str) -> None:
    doc = load(path)
    nodes, children, parent = build(doc)
    root = doc["root_id"]
    name = os.path.basename(path).replace("_graph.json", "")
    labels = collections.Counter(n["level_label"] or n["snippet_type"] for n in nodes.values())
    regions = collections.Counter(n["region"] for n in nodes.values())
    root_children = len(children.get(root, []))
    depth = depths(nodes, parent, root)
    pages = max((n.get("page_no") or 0) for n in nodes.values())
    print(f"{name[:52]:54s} n={len(nodes):5d} pages={pages:4d} depth_max={max(depth.values(), default=0):2d} "
          f"root_children={root_children:4d} ({root_children / max(len(nodes), 1):4.0%}) "
          f"head={labels.get('Heading', 0):4d} body={labels.get('Body', 0):5d} "
          f"unknown={labels.get('unknown', 0):5d} "
          f"body_region={regions.get('body', 0):5d} figure={regions.get('figure', 0):5d} "
          f"sidebar={regions.get('sidebar', 0):4d} front={regions.get('front_matter', 0):3d}")


def tree(path: str, max_text: int = 90, headings_only: bool = False,
         region: str | None = None, limit: int | None = None) -> None:
    doc = load(path)
    nodes, children, _parent = build(doc)
    root = doc["root_id"]
    print(f"# {os.path.basename(path)}")
    print(f"title: {doc.get('title')!r}")
    print(f"root:  {root[:12]} {'(synthetic)' if root not in nodes else repr(text_of(nodes[root], 60))}\n")
    printed = 0

    def walk(node_id: str, depth: int, seen: frozenset) -> None:
        nonlocal printed
        if limit is not None and printed >= limit:
            return
        node = nodes.get(node_id)
        if node is not None:
            shown = not (headings_only and node["level_label"] not in ("Title", "Heading"))
            if region and node["region"] != region:
                shown = False
            if shown:
                printed += 1
                tag = f"{node['snippet_type'][:3]}/{node['level_label'] or '-'}"
                print(f"{'  ' * depth}{tag:14s} L{node['level']:<4} p{node.get('page_no') or 0:<4}"
                      f"[{node['region'][:4]}] {text_of(node, max_text)}")
        for child_id, _weight in children.get(node_id, []):
            if child_id not in seen:
                walk(child_id, depth + 1, seen | {child_id})

    walk(root, 0, frozenset({root}))


def checks(path: str) -> None:
    doc = load(path)
    nodes, children, _parent = build(doc)
    root = doc["root_id"]
    print(f"\n=== checks: {os.path.basename(path)} ===")

    # content left on the root: every node whose parent could not be resolved lands here
    orphans = [nodes[child_id] for child_id, _ in children.get(root, []) if child_id in nodes]
    stranded = [n for n in orphans if n["level_label"] not in ("Title", "Heading")]
    print(f"[root] {len(orphans)} direct children, {len(stranded)} of them not headings")
    for node in stranded[:8]:
        print(f"    {node['snippet_type']:5} p{node.get('page_no')} [{node['region']}] {text_of(node, 80)}")

    # a subheading must rank below the heading that holds it
    inverted = []
    for head, tail, _weight in doc["edges"]:
        parent_node, child_node = nodes.get(head), nodes.get(tail)
        if not parent_node or not child_node:
            continue
        if (parent_node["level_label"] in ("Title", "Heading")
                and child_node["level_label"] in ("Title", "Heading")
                and child_node["level"] <= parent_node["level"]):
            inverted.append((parent_node, child_node))
    print(f"[levels] {len(inverted)} heading->heading edges where the child does not rank below its parent")
    for parent_node, child_node in inverted[:6]:
        print(f"    L{parent_node['level']} {text_of(parent_node, 40)!r} -> L{child_node['level']} {text_of(child_node, 40)!r}")

    longish = [n for n in nodes.values()
               if n["level_label"] == "Heading" and len(n.get("text") or "") > MAX_HEADING_CHARS]
    print(f"[headings] {len(longish)} headings longer than {MAX_HEADING_CHARS} chars")
    for node in longish[:5]:
        print(f"    p{node.get('page_no')} {text_of(node, 90)}")

    distant = sum(1 for head, tail, _w in doc["edges"]
                  if nodes.get(head) and nodes.get(tail)
                  and nodes[head].get("page_no") and nodes[tail].get("page_no")
                  and abs(nodes[tail]["page_no"] - nodes[head]["page_no"]) > MAX_EDGE_PAGE_SPAN)
    print(f"[locality] {distant} edges spanning more than {MAX_EDGE_PAGE_SPAN} pages")

    media = [n for n in nodes.values() if n["snippet_type"] in ("image", "table")]
    unreferenced = [n for n in media if n["level_label"] == "Unreferenced Image"]
    on_root = [n for n in media if n["snippet_id"] in {c for c, _ in children.get(root, [])}]
    print(f"[media] {len(media)} media nodes, {len(unreferenced)} unreferenced, {len(on_root)} left on the root")

    texts = collections.Counter(text_of(n) for n in nodes.values() if text_of(n))
    repeated = [(t, c) for t, c in texts.items() if c >= MIN_REPEATS and len(t) > MIN_REPEAT_CHARS]
    print(f"[repeats] {len(repeated)} texts occurring {MIN_REPEATS}x or more (a running header read as content?)")
    for text, count in sorted(repeated, key=lambda kv: -kv[1])[:5]:
        print(f"    x{count} {text[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("doc", nargs="?", help="substring of the document name; default: all")
    parser.add_argument("--tree", action="store_true", help="print the hierarchy")
    parser.add_argument("--headings", action="store_true", help="with --tree: the outline only")
    parser.add_argument("--region", help="with --tree: only nodes of this region")
    parser.add_argument("--checks", action="store_true", help="run the diagnostics")
    parser.add_argument("--limit", type=int, help="with --tree: stop after this many nodes")
    args = parser.parse_args()

    paths = sorted(glob.glob(f"{GRAPH_DIR}/*_graph.json"))
    if args.doc:
        paths = [p for p in paths if args.doc.lower() in os.path.basename(p).lower()]
    if not paths:
        raise SystemExit(f"no graphs under {GRAPH_DIR}/ matching {args.doc!r}; run the extractor first")

    for path in paths:
        if args.tree:
            tree(path, headings_only=args.headings, region=args.region, limit=args.limit)
        if args.checks:
            checks(path)
        if not (args.tree or args.checks):
            summary(path)


if __name__ == "__main__":
    main()
