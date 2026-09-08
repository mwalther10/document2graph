"""Write a ``DocumentGraph`` into Neo4j.

Requires the ``neo4j`` extra (``pip install document2graph[neo4j]``); the driver
is imported lazily so the rest of the graph store works without it. Both
functions take an already-constructed driver, so the caller owns the connection
and its credentials.
"""

from .document_graph import DocumentGraph

# Neo4j properties are scalars or arrays of scalars, never nested maps, so the
# bounding box and char span are flattened instead of dropped (as they were when
# this lived in the test suite).
_TEXT_PREVIEW_CHARS = 100


def ensure_neo4j_constraints(driver) -> None:
    """Create the uniqueness constraints the writes rely on.

    Every write MERGEs on ``snippet_id`` / ``document_id``; without a backing
    index that degrades to a full scan once a corpus grows past a few documents.
    """
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT snippet_id_unique IF NOT EXISTS "
            "FOR (s:Snippet) REQUIRE s.snippet_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.document_id IS UNIQUE"
        )


def _snippet_row(snippet) -> dict:
    row = {
        "snippet_id": snippet.snippet_id,
        "local_ref": snippet.local_ref,
        "document_id": snippet.document_id,
        "sequence_no": snippet.sequence_no,
        "label": snippet.label,
        "snippet_type": snippet.snippet_type,
        "level": snippet.level,
        "level_label": snippet.level_label,
        "region": snippet.region,
        "is_grouped": snippet.is_grouped,
        "page_no": snippet.page_no,
        "line_heights": snippet.line_heights,
        "font_key": snippet.font_key or "",
        "level_height": snippet.level_height if snippet.level_height is not None else -1.0,
        "text_preview": snippet.text[:_TEXT_PREVIEW_CHARS] if snippet.text else "",
        # pages this snippet spans; a paragraph stitched across a page break has several
        "pages": sorted({p.page_no for p in snippet.provenance}) or [snippet.page_no],
    }
    if snippet.bbox is not None:
        row |= {
            "bbox_l": snippet.bbox.l,
            "bbox_t": snippet.bbox.t,
            "bbox_r": snippet.bbox.r,
            "bbox_b": snippet.bbox.b,
            "bbox_origin": snippet.bbox.coord_origin.value,
        }
    if snippet.charspan is not None:
        row |= {"charspan_start": snippet.charspan[0], "charspan_end": snippet.charspan[1]}
    return row


def write_graph_to_neo4j(driver, doc_graph: DocumentGraph) -> None:
    """Upsert a document, its snippets and both edge kinds.

    Idempotent: writing the same document twice updates it in place, because
    ``document_id`` and every ``snippet_id`` derived from it are deterministic.
    """
    snippet_rows = [_snippet_row(s) for s in doc_graph.snippets]
    edge_rows = [
        {"parent": parent, "child": child, "weight": weight}
        for parent, child, weight in doc_graph.edges
    ]
    reference_rows = [
        {"source": source, "target": target, "weight": weight}
        for source, target, weight in doc_graph.reference_edges
    ]
    with driver.session() as session:
        session.run(
            "MERGE (d:Document {document_id: $document_id}) "
            "SET d.filename = $filename, d.title = $title, "
            "    d.document_type = $document_type, d.version = $version, "
            "    d.authors = $authors, d.institutions = $institutions",
            document_id=doc_graph.document_id,
            filename=doc_graph.filename,
            title=doc_graph.title,
            document_type=doc_graph.document_type,
            version=doc_graph.version,
            authors=doc_graph.authors,
            institutions=doc_graph.institutions,
        )
        session.run(
            "UNWIND $rows AS row "
            "MERGE (s:Snippet {snippet_id: row.snippet_id}) "
            "SET s += row",
            rows=snippet_rows,
        )
        session.run(
            "MATCH (d:Document {document_id: $document_id}) "
            "MATCH (r:Snippet {snippet_id: $root_id}) "
            "MERGE (d)-[:HAS_ROOT]->(r)",
            document_id=doc_graph.document_id,
            root_id=doc_graph.root_id,
        )
        session.run(
            "UNWIND $rows AS row "
            "MATCH (p:Snippet {snippet_id: row.parent}) "
            "MATCH (c:Snippet {snippet_id: row.child}) "
            "MERGE (p)-[e:HAS_CHILD]->(c) "
            "SET e.weight = row.weight",
            rows=edge_rows,
        )
        session.run(
            "UNWIND $rows AS row "
            "MATCH (t:Snippet {snippet_id: row.source}) "
            "MATCH (m:Snippet {snippet_id: row.target}) "
            "MERGE (t)-[e:REFERENCES]->(m) "
            "SET e.weight = row.weight",
            rows=reference_rows,
        )
