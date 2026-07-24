from types import SimpleNamespace

import networkx as nx
import pytest
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ListItem, ProvenanceItem, RefItem, TextItem

from document2graph.document2graph_extractor.snippet_graph_constructor import (
    CAPTION_PATTERN,
    ROOT_NODE_ID,
    SnippetGraphConstructor,
)
from document2graph.utils.log import Log
from document2graph.models import (
    Document,
    EdgeWeightConfig,
    ImageSnippetNode,
    MetadataExtractionConfig,
    TextSnippet,
    TextSnippetNode,
)

PAGE_WIDTH, PAGE_HEIGHT = 600.0, 800.0
# a drawn sidebar box (l, r, b, t) in the body block of the right column
SIDEBAR_BOX = (310.0, 560.0, 100.0, 280.0)


def make_text_node(idx: int, level: int, level_label: str, parent_id: str | None, is_grouped: bool = False, region: str = "body", group_ref: str = "#/body") -> TextSnippetNode:
    return TextSnippetNode(
        snippet_id=f"#/texts/{idx}",
        document_id="doc-1",
        label="text",
        sequence_no=idx,
        level=level,
        level_label=level_label,
        parent_id=parent_id,
        docling_parent_ref=RefItem(**{"$ref": group_ref}),
        docling_self_ref=RefItem(**{"$ref": f"#/texts/{idx}"}),
        is_grouped=is_grouped,
        region=region,
        text=f"text {idx}",
        bbox=BoundingBox(l=0, t=1, r=1, b=0),
        charspan=(0, 6),
        page_no=1,
    )


def make_snippet(idx: int, top: float, left: float = 50.0, page_no: int = 1, text: str | None = None,
                 label: str = "text") -> TextSnippet:
    """A text snippet placed on a page (bottom-left origin, as docling reports it)."""
    text = f"text {idx}" if text is None else text
    # docling models a bullet as its own item type, which is what makes its own merge
    # rule skip these fragments
    item_type = ListItem if label == "list_item" else TextItem
    item = item_type(
        self_ref=f"#/texts/{idx}", label=label, orig=text, text=text,
        prov=[ProvenanceItem(
            page_no=page_no, charspan=(0, len(text)),
            bbox=BoundingBox(l=left, t=top, r=left + 200, b=top - 10, coord_origin=CoordOrigin.BOTTOMLEFT),
        )],
    )
    return TextSnippet(text_item=item)


def make_page(*rule_tops: float, boxes: tuple[tuple[float, float, float, float], ...] = ()) -> SimpleNamespace:
    """A page carrying one full-width horizontal rule per given y position, plus any
    drawn boxes as (l, r, b, t)."""
    shapes = [
        SimpleNamespace(
            points=[(30.0, top), (PAGE_WIDTH - 30.0, top), (PAGE_WIDTH - 30.0, top - 0.5), (30.0, top - 0.5)],
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        for top in rule_tops
    ]
    shapes += [
        SimpleNamespace(
            points=[(left, bottom), (right, bottom), (right, top), (left, top)],
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        for left, right, bottom, top in boxes
    ]
    return SimpleNamespace(shapes=shapes, dimension=SimpleNamespace(width=PAGE_WIDTH, height=PAGE_HEIGHT))


@pytest.fixture
def paged_constructor() -> SnippetGraphConstructor:
    """Constructor over a single page ruled into a title, a front matter and a body block,
    with a sidebar box drawn in the lower half of the right column."""
    c = SnippetGraphConstructor.__new__(SnippetGraphConstructor)
    c._page_shapes_cache = {}
    c.metadata_config = MetadataExtractionConfig(title_page=1)
    c.logger = Log("test").logger
    c.pdf_doc = SimpleNamespace(
        get_page=lambda page_no: make_page(760.0, 700.0, 300.0, boxes=(SIDEBAR_BOX,))
    )
    return c


def make_image_node(idx: int, level_label: str, parent_id: str) -> ImageSnippetNode:
    return ImageSnippetNode(
        snippet_id=f"#/pictures/{idx}",
        document_id="doc-1",
        label="picture",
        sequence_no=idx,
        level=2,
        level_label=level_label,
        parent_id=parent_id,
        docling_parent_ref=None,
        docling_self_ref=None,
        caption_text="a caption",
        bbox=BoundingBox(l=0, t=1, r=1, b=0),
        page_no=1,
    )


@pytest.fixture
def constructor() -> SnippetGraphConstructor:
    # bypass __init__: these methods only need edge_weights and document metadata
    c = SnippetGraphConstructor.__new__(SnippetGraphConstructor)
    c.edge_weights = EdgeWeightConfig()
    c._document_metadata = Document(document_id="doc-1", title="Test Doc")
    return c


def test_edge_weights_by_category(constructor: SnippetGraphConstructor):
    weights = constructor.edge_weights
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    subheading = make_text_node(1, level=1, level_label="Heading", parent_id=heading.snippet_id)
    body = make_text_node(2, level=2, level_label="Body", parent_id=subheading.snippet_id)
    bullet = make_text_node(3, level=2, level_label="Body", parent_id=body.snippet_id, is_grouped=True)
    image = make_image_node(0, level_label="Body", parent_id=body.snippet_id)
    orphan_image = make_image_node(1, level_label="Unreferenced Image", parent_id=heading.snippet_id)

    assert constructor.compute_edge_weight(heading, subheading) == weights.section
    assert constructor.compute_edge_weight(subheading, body) == weights.text
    assert constructor.compute_edge_weight(body, bullet) == weights.list_item
    assert constructor.compute_edge_weight(body, image) == weights.media
    assert constructor.compute_edge_weight(heading, orphan_image) == weights.unreferenced_media


def test_custom_edge_weights_are_used(constructor: SnippetGraphConstructor):
    constructor.edge_weights = EdgeWeightConfig(section=5.0, text=2.5)
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    subheading = make_text_node(1, level=1, level_label="Heading", parent_id=heading.snippet_id)
    body = make_text_node(2, level=2, level_label="Body", parent_id=subheading.snippet_id)

    assert constructor.compute_edge_weight(heading, subheading) == 5.0
    assert constructor.compute_edge_weight(subheading, body) == 2.5


def test_orphans_and_dangling_parents_attach_to_root(constructor: SnippetGraphConstructor):
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    body = make_text_node(1, level=2, level_label="Body", parent_id=heading.snippet_id)
    dangling = make_text_node(2, level=2, level_label="Body", parent_id="#/does/not/exist")
    nodes = [heading, body, dangling]

    edges = constructor.construct_snippet_edges(nodes, root_id=ROOT_NODE_ID)

    assert (ROOT_NODE_ID, heading.snippet_id, constructor.edge_weights.root) in edges
    assert (heading.snippet_id, body.snippet_id, constructor.edge_weights.text) in edges
    assert (ROOT_NODE_ID, dangling.snippet_id, constructor.edge_weights.root) in edges
    # every node has exactly one incoming edge
    assert len(edges) == len(nodes)


def test_page_blocks_are_read_one_after_the_other(paged_constructor: SnippetGraphConstructor):
    """Docling reads a two-column page column by column, so the right-hand continuation
    of the front matter (600) arrives after the body text that follows it in the left
    column (250). Both columns of a block belong together and are read first."""
    left_front = make_snippet(0, top=650.0, left=50.0)
    left_body = make_snippet(1, top=250.0, left=50.0)
    right_front = make_snippet(2, top=600.0, left=320.0)
    right_body = make_snippet(3, top=200.0, left=320.0)

    ordered = paged_constructor.order_by_page_blocks([left_front, left_body, right_front, right_body])

    assert [s.text_item.self_ref for s in ordered] == ["#/texts/0", "#/texts/2", "#/texts/1", "#/texts/3"]


def test_snippets_of_later_pages_keep_their_order(paged_constructor: SnippetGraphConstructor):
    first_page = make_snippet(0, top=650.0)
    later = [make_snippet(i, top=650.0 - 10 * i, page_no=2) for i in range(1, 4)]

    ordered = paged_constructor.order_by_page_blocks([first_page, *later])

    assert [s.text_item.self_ref for s in ordered] == [f"#/texts/{i}" for i in range(4)]


def test_front_matter_is_the_block_above_the_lowest_rule(paged_constructor: SnippetGraphConstructor):
    title = make_snippet(0, top=730.0)          # between the first two rules
    front = make_snippet(1, top=650.0)          # last block above the lowest rule
    body = make_snippet(2, top=250.0)           # below the lowest rule
    second_page = make_snippet(3, top=650.0, page_no=2)

    front_matter = paged_constructor.find_front_matter([title, front, body, second_page])

    assert front_matter == {"#/texts/1"}


def test_nothing_is_front_matter_without_a_body_below_the_rule(paged_constructor: SnippetGraphConstructor):
    """A rule with no content under it separates nothing (a footer rule, say)."""
    assert paged_constructor.find_front_matter([make_snippet(0, top=650.0)]) == set()


def test_paragraph_cut_at_a_column_break_is_stitched(paged_constructor: SnippetGraphConstructor):
    """docling ends the left column with a list item and labels the continuation at the
    top of the right column as plain text, so it never merges the two itself."""
    head = make_snippet(0, top=340.0, left=50.0, label="list_item",
                        text="Vermeidung von Akut- und Folgekom-")
    tail = make_snippet(1, top=690.0, left=320.0, text="plikationen, weniger Schmerzen.")

    stitched = paged_constructor.stitch_continuations([head, tail])

    assert len(stitched) == 1
    assert stitched[0].text_item.text == "Vermeidung von Akut- und Folgekomplikationen, weniger Schmerzen."
    assert stitched[0].text_item.label == "list_item"          # the opening fragment wins
    assert stitched[0].text_item.prov[0] == head.text_item.prov[0]  # primary bbox kept
    assert [p.page_no for p in stitched[0].text_item.prov] == [1, 1]


def test_stitching_joins_whole_words_with_a_space(paged_constructor: SnippetGraphConstructor):
    head = make_snippet(0, top=340.0, left=50.0, text="können in der ambulanten und")
    tail = make_snippet(1, top=690.0, left=320.0, text="stationären Pflege eingesetzt werden.")

    stitched = paged_constructor.stitch_continuations([head, tail])

    assert stitched[0].text_item.text == "können in der ambulanten und stationären Pflege eingesetzt werden."


def test_suspended_hyphen_survives_stitching(paged_constructor: SnippetGraphConstructor):
    """"Akut- und Folgekomplikationen": the hyphen stands for a dropped word part and
    must not glue the two fragments together."""
    head = make_snippet(0, top=340.0, left=50.0, text="Vermeidung von Akut-")
    tail = make_snippet(1, top=690.0, left=320.0, text="und Folgekomplikationen.")

    stitched = paged_constructor.stitch_continuations([head, tail])

    assert stitched[0].text_item.text == "Vermeidung von Akut- und Folgekomplikationen."


def test_finished_sentence_is_not_stitched(paged_constructor: SnippetGraphConstructor):
    head = make_snippet(0, top=340.0, left=50.0, text="Der Satz ist zu Ende.")
    tail = make_snippet(1, top=690.0, left=320.0, text="der nächste beginnt klein")

    assert len(paged_constructor.stitch_continuations([head, tail])) == 2


def test_column_neighbours_away_from_the_column_ends_are_left_alone(paged_constructor: SnippetGraphConstructor):
    """A two-column glossary reads term, definition, term, ... Both fragments look
    unfinished, but neither sits at the end of its column, so they stay apart."""
    head = make_snippet(0, top=400.0, left=50.0, text="kontinuierliche subkutane Insulininfusion")
    tail = make_snippet(1, top=380.0, left=320.0, text="freies Thyroxin")
    below_head = make_snippet(2, top=340.0, left=50.0, text="Weiterer Eintrag der linken Spalte.")
    above_tail = make_snippet(3, top=690.0, left=320.0, text="Kopf der rechten Spalte.")

    stitched = paged_constructor.stitch_continuations([above_tail, head, tail, below_head])

    assert len(stitched) == 4


def test_stitching_does_not_cross_the_front_matter_boundary(paged_constructor: SnippetGraphConstructor):
    """The masthead and the body are separate blocks: text never flows from one to the
    other, however unfinished the last masthead line looks."""
    masthead = make_snippet(0, top=650.0, left=320.0, text="Deutsche Gesellschaft für Innere Medizin")
    body = make_snippet(1, top=250.0, left=50.0, text="stationäre Versorgung von Menschen mit Diabetes.")

    assert len(paged_constructor.stitch_continuations([masthead, body])) == 2


def test_text_inside_a_drawn_box_is_a_sidebar(paged_constructor: SnippetGraphConstructor):
    """The recommendation boxes of the guideline layout are drawn rectangles; docling
    labels their titles as section headers, but they are asides, not sections."""
    boxed = make_snippet(0, top=250.0, left=320.0, text="EMPFEHLUNGEN")
    beside = make_snippet(1, top=250.0, left=50.0, text="Therapie")

    regions = {s.text_item.self_ref: s.region for s in paged_constructor.assign_regions([boxed, beside])}

    assert regions == {"#/texts/0": "sidebar", "#/texts/1": "body"}


def test_front_matter_and_figures_outrank_the_box_test(paged_constructor: SnippetGraphConstructor):
    """A snippet is classified by the most specific region it belongs to."""
    in_figure = make_snippet(0, top=250.0, left=320.0, text="Abb. 1")
    in_figure.text_item.parent = RefItem(**{"$ref": "#/pictures/0"})
    masthead = make_snippet(1, top=650.0, left=50.0, text="Institute")

    regions = {s.text_item.self_ref: s.region for s in paged_constructor.assign_regions([in_figure, masthead])}

    assert regions == {"#/texts/0": "figure", "#/texts/1": "front_matter"}


def test_sidebars_hang_off_the_section_they_sit_in(constructor: SnippetGraphConstructor):
    """An aside belongs to the section it is printed in, but never holds one."""
    section = make_text_node(0, level=1, level_label="Heading", parent_id=None)
    box_title = make_text_node(1, level=3, level_label="Heading", parent_id=None, region="sidebar")
    box_text = make_text_node(2, level=4, level_label="Body", parent_id=None, region="sidebar")
    later_section = make_text_node(3, level=1, level_label="Heading", parent_id=None)
    subsection = make_text_node(4, level=2, level_label="Heading", parent_id=None)
    nodes = [section, box_title, box_text, later_section, subsection]

    assert constructor.compute_parent_id(box_title, nodes) == section.snippet_id
    assert constructor.compute_parent_id(box_text, nodes) == box_title.snippet_id
    # the subsection skips the box and attaches to the section
    assert constructor.compute_parent_id(subsection, nodes) == later_section.snippet_id


def test_caption_pattern_reads_past_a_typographic_marker():
    """Docling keeps the marker that introduces a caption in the text, and the guideline
    layout sets every one of them that way -- only one table of 054 came out captioned."""
    assert CAPTION_PATTERN.match("▶ Tab.1 Therapieziele. Quelle: DDG")
    assert CAPTION_PATTERN.match("Tab.2 Mortalität")
    assert CAPTION_PATTERN.match("▪ Fig. 2b")
    # a mention inside a sentence is not a caption
    assert not CAPTION_PATTERN.match("Der Tab.1 Verweis im Fließtext")
    assert not CAPTION_PATTERN.match("Tabelle ohne Nummer")


def test_word_cells_over_two_rows_yield_the_row_pitch(constructor: SnippetGraphConstructor):
    """Fonts with a broken glyph box also make docling split a line into one cell per
    word; counting cells would divide the heading by its word count."""
    row_a = [SimpleNamespace(rect=SimpleNamespace(r_y0=700.0, r_y3=701.9)) for _ in range(4)]
    row_b = [SimpleNamespace(rect=SimpleNamespace(r_y0=690.0, r_y3=691.9)) for _ in range(3)]

    assert constructor.count_text_rows(row_a + row_b) == 2
    assert constructor.count_text_rows(row_a) == 1
    assert constructor.count_text_rows([]) == 0


def test_front_matter_does_not_parent_body_snippets(constructor: SnippetGraphConstructor):
    """The body starts a new block: it hangs off the title, never off the masthead."""
    title = make_text_node(0, level=0, level_label="Title", parent_id=None)
    front_heading = make_text_node(1, level=1, level_label="Heading", parent_id=None, region="front_matter")
    front_text = make_text_node(2, level=2, level_label="Body", parent_id=None, region="front_matter")
    body_text = make_text_node(3, level=2, level_label="Body", parent_id=None)
    nodes = [title, front_heading, front_text, body_text]

    assert constructor.compute_parent_id(front_text, nodes) == front_heading.snippet_id
    assert constructor.compute_parent_id(body_text, nodes) == title.snippet_id


def test_list_split_across_blocks_attaches_per_block(constructor: SnippetGraphConstructor):
    """A list continued in the next column can share a docling group with the body list
    that follows it in the raw reading order; each part attaches within its own block."""
    front_heading = make_text_node(0, level=1, level_label="Heading", parent_id=None, region="front_matter")
    front_bullet = make_text_node(1, level=2, level_label="Body", parent_id=None, is_grouped=True,
                                  region="front_matter", group_ref="#/groups/2")
    body_text = make_text_node(2, level=2, level_label="Body", parent_id=None)
    body_bullet = make_text_node(3, level=2, level_label="Body", parent_id=None, is_grouped=True,
                                 group_ref="#/groups/2")
    nodes = [front_heading, front_bullet, body_text, body_bullet]

    assert constructor.compute_parent_id(front_bullet, nodes) == front_heading.snippet_id
    assert constructor.compute_parent_id(body_bullet, nodes) == body_text.snippet_id


def test_list_opening_the_body_block_attaches_to_the_title(constructor: SnippetGraphConstructor):
    """Nothing of the body precedes the list, so it hangs off the title rather than off
    the last line of the masthead."""
    title = make_text_node(0, level=0, level_label="Title", parent_id=None)
    front_bullet = make_text_node(1, level=2, level_label="Body", parent_id=None, is_grouped=True,
                                  region="front_matter", group_ref="#/groups/1")
    body_bullet = make_text_node(2, level=2, level_label="Body", parent_id=None, is_grouped=True,
                                 group_ref="#/groups/2")
    nodes = [title, front_bullet, body_bullet]

    assert constructor.compute_parent_id(body_bullet, nodes) == title.snippet_id


def test_graph_is_connected(constructor: SnippetGraphConstructor):
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    body = make_text_node(1, level=2, level_label="Body", parent_id=heading.snippet_id)
    # cycle detached from the root: 2 -> 3 -> 2
    cycle_a = make_text_node(2, level=1, level_label="Heading", parent_id="#/texts/3")
    cycle_b = make_text_node(3, level=2, level_label="Body", parent_id="#/texts/2")
    nodes = [heading, body, cycle_a, cycle_b]

    edges = constructor.construct_snippet_edges(nodes, root_id=ROOT_NODE_ID)
    edges += constructor.connect_components(nodes, edges, root_id=ROOT_NODE_ID)
    graph = constructor.build_nx_graph([heading, body, cycle_a, cycle_b], [], [], edges)

    assert nx.is_weakly_connected(graph)
    # the cycle is attached via its structurally highest node
    assert graph.has_edge(ROOT_NODE_ID, cycle_a.snippet_id)
    assert all("weight" in data for _, _, data in graph.edges(data=True))
