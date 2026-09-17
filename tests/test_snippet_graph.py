from collections import OrderedDict
from types import SimpleNamespace

import networkx as nx
import pytest
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell, TextCellUnit
from docling_core.types.doc.document import (
    ListItem,
    ProvenanceItem,
    RefItem,
    TableCell,
    TableData,
    TableItem,
    TextItem,
)

from document2graph.document2graph_extractor.snippet_graph_constructor import (
    CAPTION_PATTERN,
    PAGE_CELL_CACHE_PAGES,
    ROOT_NODE_ID,
    SnippetGraphConstructor,
)
from document2graph.utils.log import Log
from document2graph.models import (
    Document,
    EdgeWeightConfig,
    ImageSnippetNode,
    MetadataExtractionConfig,
    PipelineFlags,
    TextSnippet,
    TextSnippetNode,
)

PAGE_WIDTH, PAGE_HEIGHT = 600.0, 800.0
# a drawn sidebar box (l, r, b, t) in the body block of the right column
SIDEBAR_BOX = (310.0, 560.0, 100.0, 280.0)


def make_text_node(idx: int, level: int, level_label: str, parent_id: str | None, is_grouped: bool = False, region: str = "body", group_ref: str = "#/body", top: float = 1.0, left: float = 50.0, page_no: int = 1) -> TextSnippetNode:
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
        bbox=BoundingBox(l=left, t=top, r=left + 200, b=top - 10, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 6),
        page_no=page_no,
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


def make_image_node(idx: int, level_label: str, parent_id: str | None, caption_text: str = "a caption", top: float = 1.0, left: float = 50.0, page_no: int = 1) -> ImageSnippetNode:
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
        caption_text=caption_text,
        bbox=BoundingBox(l=left, t=top, r=left + 200, b=top - 10, coord_origin=CoordOrigin.BOTTOMLEFT),
        page_no=page_no,
    )


@pytest.fixture
def constructor() -> SnippetGraphConstructor:
    # bypass __init__: these methods only need edge_weights, document metadata and
    # the relation map the edge builders record into
    c = SnippetGraphConstructor.__new__(SnippetGraphConstructor)
    c.edge_weights = EdgeWeightConfig()
    c._edge_relations = {}
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


def test_edge_relations_by_category(constructor: SnippetGraphConstructor):
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    subheading = make_text_node(1, level=1, level_label="Heading", parent_id=heading.snippet_id)
    body = make_text_node(2, level=2, level_label="Body", parent_id=subheading.snippet_id)
    bullet = make_text_node(3, level=2, level_label="Body", parent_id=body.snippet_id, is_grouped=True)
    image = make_image_node(0, level_label="Body", parent_id=body.snippet_id)
    orphan_image = make_image_node(1, level_label="Unreferenced Image", parent_id=heading.snippet_id)

    assert constructor.edge_relation(heading, subheading) == "section"
    assert constructor.edge_relation(subheading, body) == "text"
    assert constructor.edge_relation(body, bullet) == "list_item"
    assert constructor.edge_relation(body, image) == "media"
    assert constructor.edge_relation(heading, orphan_image) == "unreferenced_media"


def test_constructing_edges_records_their_relation(constructor: SnippetGraphConstructor):
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    bullet = make_text_node(1, level=1, level_label="Body", parent_id=heading.snippet_id,
                            is_grouped=True)
    orphan = make_text_node(2, level=1, level_label="Body", parent_id="#/texts/999")

    constructor.construct_snippet_edges([heading, bullet, orphan], heading.snippet_id)

    assert constructor._edge_relations[(heading.snippet_id, bullet.snippet_id)] == "list_item"
    # a dangling parent_id attaches to the root, and says so
    assert constructor._edge_relations[(heading.snippet_id, orphan.snippet_id)] == "root"


def test_edge_context_carries_relations_and_unit_kinds(constructor: SnippetGraphConstructor):
    heading = make_text_node(0, level=0, level_label="Heading", parent_id=None)
    bullet = make_text_node(1, level=1, level_label="Body", parent_id=heading.snippet_id,
                            is_grouped=True)
    image = make_image_node(0, level_label="Body", parent_id=heading.snippet_id)
    nodes = [heading, bullet, image]
    edges = constructor.construct_snippet_edges(nodes, heading.snippet_id)

    ctx = constructor.build_edge_context(nodes, heading.snippet_id, [(p, c) for p, c, _ in edges])

    assert ctx.relation((heading.snippet_id, bullet.snippet_id)) == "list_item"
    assert ctx.unit(bullet.snippet_id).is_grouped
    assert ctx.unit(image.snippet_id).kind == "image"
    assert ctx.text(bullet.snippet_id) == bullet.text


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


def test_page_background_rectangle_is_not_a_sidebar(paged_constructor: SnippetGraphConstructor):
    """Word processors draw a rectangle the size of the page under every page. Read as a
    box it puts the whole document inside a sidebar, leaving no body text to rank the
    headings into an outline."""
    paged_constructor._page_shapes_cache = {}
    paged_constructor.pdf_doc = SimpleNamespace(
        get_page=lambda page_no: make_page(
            760.0, 700.0, 300.0, boxes=((0.0, PAGE_WIDTH, 0.0, PAGE_HEIGHT), SIDEBAR_BOX)
        )
    )
    boxed = make_snippet(0, top=250.0, left=320.0, text="EMPFEHLUNGEN")
    beside = make_snippet(1, top=250.0, left=50.0, text="Therapie")

    assert paged_constructor.page_boxes(1) == [SIDEBAR_BOX]
    regions = {s.text_item.self_ref: s.region for s in paged_constructor.assign_regions([boxed, beside])}
    assert regions == {"#/texts/0": "sidebar", "#/texts/1": "body"}


def test_unreferenced_media_attaches_to_the_closest_heading_above_it(paged_constructor: SnippetGraphConstructor):
    """A table whose caption references nothing belongs to the section it is printed in.
    Media nodes are numbered within the media list, so the heading above them has to be
    found by position on the page, not by sequence_no."""
    paged_constructor.levels = SimpleNamespace(header_levels=lambda: {0, 1})
    title = make_text_node(0, level=0, level_label="Title", parent_id=None, top=760.0)
    early = make_text_node(1, level=1, level_label="Heading", parent_id=title.snippet_id, top=280.0)
    closest = make_text_node(2, level=1, level_label="Heading", parent_id=title.snippet_id, top=200.0)
    below = make_text_node(3, level=1, level_label="Heading", parent_id=title.snippet_id, top=100.0)
    table = make_image_node(0, level_label="", parent_id=None, caption_text="", top=150.0)

    [placed] = paged_constructor.rb_image_parent_matching([table], [title, early, closest, below])

    assert placed.parent_id == closest.snippet_id
    assert placed.level == closest.level + 1
    assert placed.level_label == "Unreferenced Image"


def test_unreferenced_media_skips_aside_headings(paged_constructor: SnippetGraphConstructor):
    """A figure label or the title of a sidebar box heads no section, so a table printed
    past one still belongs to the body section around it."""
    paged_constructor.levels = SimpleNamespace(header_levels=lambda: {0, 1, 2})
    section = make_text_node(0, level=1, level_label="Heading", parent_id=None, top=280.0)
    box_title = make_text_node(1, level=2, level_label="Heading", parent_id=None, top=200.0, region="sidebar")
    table = make_image_node(0, level_label="", parent_id=None, caption_text="", top=150.0)

    [placed] = paged_constructor.rb_image_parent_matching([table], [section, box_title])

    assert placed.parent_id == section.snippet_id


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


def flagged_constructor(paged: SnippetGraphConstructor, **flags) -> SnippetGraphConstructor:
    """The paged constructor with the ablation flags set and the per-snippet
    typography lookups stubbed out (they need a real parsed page)."""
    paged.flags = PipelineFlags(**flags)
    paged.add_line_heights = lambda item: [10.0]
    paged.get_font_key = lambda item: "F1"
    return paged


def make_docling_doc(*snippets: TextSnippet) -> SimpleNamespace:
    # `pages` mirrors DoclingDocument's own: page_no -> item carrying the page
    # box. It is what page_geometry() reads, so a fake without it would make
    # get_graph() fail on the double rather than on anything under test.
    pages = {
        page_no: SimpleNamespace(
            page_no=page_no, size=SimpleNamespace(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        )
        for page_no in sorted({s.text_item.prov[0].page_no for s in snippets} or {1})
    }
    return SimpleNamespace(
        texts=[s.text_item for s in snippets], tables=[], pictures=[], pages=pages
    )


def test_stitching_flag_off_leaves_the_column_fragments_apart(paged_constructor: SnippetGraphConstructor):
    head = make_snippet(0, top=340.0, left=50.0, text="können in der ambulanten und")
    tail = make_snippet(1, top=690.0, left=320.0, text="stationären Pflege eingesetzt werden.")
    doc = make_docling_doc(head, tail)

    on = flagged_constructor(paged_constructor, stitch_continuations=True)._prepare_text_items(doc)
    assert len(on) == 1

    off = flagged_constructor(paged_constructor, stitch_continuations=False)._prepare_text_items(doc)
    assert len(off) == 2
    assert [s.text_item.text for s in off] == [head.text_item.text, tail.text_item.text]


def test_reading_order_flag_off_keeps_doclings_order(paged_constructor: SnippetGraphConstructor):
    """docling reads the page column by column, so the left column's body text arrives
    before the right column's continuation of the block above it. The repair puts the
    blocks back in order; turning it off leaves docling's order alone."""
    left_body = make_snippet(0, top=250.0, left=50.0, text="Linke Spalte, unterer Block.")
    right_front = make_snippet(1, top=600.0, left=320.0, text="Rechte Spalte, oberer Block.")
    doc = make_docling_doc(left_body, right_front)

    on = flagged_constructor(paged_constructor, repair_reading_order=True, stitch_continuations=False,
                             assign_regions=False)._prepare_text_items(doc)
    assert [s.text_item.self_ref for s in on] == ["#/texts/1", "#/texts/0"]

    off = flagged_constructor(paged_constructor, repair_reading_order=False, stitch_continuations=False,
                              assign_regions=False)._prepare_text_items(doc)
    assert [s.text_item.self_ref for s in off] == ["#/texts/0", "#/texts/1"]


def test_region_flag_off_leaves_everything_in_the_body(paged_constructor: SnippetGraphConstructor):
    """With regions off the masthead and a boxed sidebar are ordinary body text, so
    they take part in the document outline."""
    masthead = make_snippet(0, top=650.0, left=50.0, text="Deutsche Gesellschaft")
    boxed = make_snippet(1, top=250.0, left=320.0, text="EMPFEHLUNGEN")
    doc = make_docling_doc(masthead, boxed)

    on = flagged_constructor(paged_constructor, assign_regions=True,
                             stitch_continuations=False)._prepare_text_items(doc)
    assert {s.region for s in on} == {"front_matter", "sidebar"}

    off = flagged_constructor(paged_constructor, assign_regions=False,
                              stitch_continuations=False)._prepare_text_items(doc)
    assert {s.region for s in off} == {"body"}


# --------------------------------------------------------- line fragment merging

def sidebar_snippet(idx: int, top: float, text: str, left: float = 320.0) -> TextSnippet:
    """A fragment inside the drawn sidebar box of the paged fixture."""
    return make_snippet(idx, top=top, left=left, text=text).model_copy(update={"region": "sidebar"})


def test_lines_of_one_sidebar_block_are_merged(paged_constructor: SnippetGraphConstructor):
    """docling reads a boxed sidebar line by line, so one bullet arrives as three
    snippets, each cut mid-sentence."""
    lines = [
        sidebar_snippet(0, top=250.0, text="nach 8 bis 12 Stunden Nahrungs-, Nikotin-"),
        sidebar_snippet(1, top=238.0, text="und Alkoholkarenz, im Sitzen oder Liegen,"),
        sidebar_snippet(2, top=226.0, text="ohne Muskelanstrengung."),
    ]

    merged = paged_constructor.merge_line_fragments(lines)

    assert len(merged) == 1
    assert merged[0].text_item.text == (
        "nach 8 bis 12 Stunden Nahrungs-, Nikotin- und Alkoholkarenz, "
        "im Sitzen oder Liegen, ohne Muskelanstrengung."
    )
    assert [p.page_no for p in merged[0].text_item.prov] == [1, 1, 1]


def test_body_text_is_never_line_merged(paged_constructor: SnippetGraphConstructor):
    """Two bullets of an enumeration both end on a comma and sit one line apart. In the
    body docling's own clustering is right, so they stay two list items."""
    first = make_snippet(0, top=250.0, left=50.0, label="list_item",
                         text="SIDD (schwerer insulindefizienter Diabetes),")
    second = make_snippet(1, top=238.0, left=50.0, label="list_item",
                          text="SIRD (schwerer insulinresistenter Diabetes),")

    assert len(paged_constructor.merge_line_fragments([first, second])) == 2


def test_fragments_a_paragraph_apart_are_left_alone(paged_constructor: SnippetGraphConstructor):
    """Twelve points below is a new block, not the next line of this one."""
    head = sidebar_snippet(0, top=250.0, text="Neuerung 1: Einzelne Aktualisierungen,")
    far = sidebar_snippet(1, top=225.0, text="Begründung: neue Eigenschaften der Systeme.")

    assert len(paged_constructor.merge_line_fragments([head, far])) == 2


def test_side_by_side_fragments_are_left_alone(paged_constructor: SnippetGraphConstructor):
    """Two labels printed beside each other in a figure are vertically adjacent but
    share no horizontal extent, so they are not one running text."""
    left = sidebar_snippet(0, top=250.0, text="Grenzen der Betriebs-", left=320.0)
    right = sidebar_snippet(1, top=240.0, text="bedingungen beachten", left=530.0)

    assert len(paged_constructor.merge_line_fragments([left, right])) == 2


def test_line_merging_flag_off_leaves_the_fragments_apart(paged_constructor: SnippetGraphConstructor):
    boxed = [make_snippet(0, top=250.0, left=320.0, text="nach 8 bis 12 Stunden Nahrungs-,"),
             make_snippet(1, top=238.0, left=320.0, text="Nikotin- und Alkoholkarenz.")]
    doc = make_docling_doc(*boxed)

    on = flagged_constructor(paged_constructor, stitch_continuations=False,
                             merge_line_fragments=True)._prepare_text_items(doc)
    assert len(on) == 1

    off = flagged_constructor(paged_constructor, stitch_continuations=False,
                              merge_line_fragments=False)._prepare_text_items(doc)
    assert len(off) == 2


# ------------------------------------------------------ table continuation merging

def make_table(idx: int, rows: list[list[str]], page_no: int, top: float, bottom: float,
               left: float = 50.0, right: float = 550.0, header: bool = True) -> TableItem:
    """A table item as docling reports it, laid out on a page."""
    cells = [
        TableCell(
            text=text, row_span=1, col_span=1,
            start_row_offset_idx=r, end_row_offset_idx=r + 1,
            start_col_offset_idx=c, end_col_offset_idx=c + 1,
            column_header=header and r == 0,
        )
        for r, row in enumerate(rows) for c, text in enumerate(row)
    ]
    return TableItem(
        self_ref=f"#/tables/{idx}", label="table",
        data=TableData(table_cells=cells, num_rows=len(rows), num_cols=len(rows[0])),
        prov=[ProvenanceItem(
            page_no=page_no, charspan=(0, 0),
            bbox=BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.BOTTOMLEFT),
        )],
    )


def test_table_continued_on_the_next_page_is_merged(paged_constructor: SnippetGraphConstructor):
    """The Teststreifenbedarf table runs off the bottom of one page and resumes at the
    top of the next, repeating its row labels."""
    head = make_table(0, [["Teststreifenbedarf", ">4 taglich"], ["Messintervall", "taglich"]],
                      page_no=1, top=730.0, bottom=60.0)
    tail = make_table(1, [["Teststreifenbedarf", ">2 taglich"], ["Messintervall", "woechentlich"]],
                      page_no=2, top=730.0, bottom=400.0)

    merged = paged_constructor.merge_table_continuations([head, tail])

    assert len(merged) == 1
    assert merged[0].self_ref == "#/tables/0"
    assert merged[0].data.num_rows == 4
    assert [p.page_no for p in merged[0].prov] == [1, 2]
    assert paged_constructor._table_continuations == {"#/tables/0": ["#/tables/1"]}


def test_merged_table_keeps_both_halves_in_its_grid(paged_constructor: SnippetGraphConstructor):
    head = make_table(0, [["AID", "Automated Insulin Delivery"], ["CGM", "Continuous Glucose Monitoring"]],
                      page_no=1, top=730.0, bottom=60.0, header=False)
    tail = make_table(1, [["IQWiG", "Institut fuer Qualitaet"], ["ISF", "Insulinsensitivitaetsfaktor"]],
                      page_no=2, top=730.0, bottom=400.0, header=False)

    merged = paged_constructor.merge_table_continuations([head, tail])[0]

    assert [row[0].text for row in merged.data.grid] == ["AID", "CGM", "IQWiG", "ISF"]
    # the continuation's rows never become a second header
    assert not any(cell.column_header for row in merged.data.grid[2:] for cell in row)


def test_sibling_boxes_sharing_a_header_are_not_merged(paged_constructor: SnippetGraphConstructor):
    """Every recommendation of the guideline layout is its own two-row table headed
    "Empfehlungen". A repeated header row alone must not join them."""
    first = make_table(0, [["Empfehlungen", "Empfehlungsgrad"], ["Ein individueller Zielwert ...", "A"]],
                       page_no=1, top=300.0, bottom=60.0)
    second = make_table(1, [["Empfehlungen", "Empfehlungsgrad"], ["Die Therapie soll ...", "B"]],
                        page_no=2, top=730.0, bottom=600.0)

    assert len(paged_constructor.merge_table_continuations([first, second])) == 2


def test_table_continued_across_a_column_break_is_not_merged(paged_constructor: SnippetGraphConstructor):
    """Only page breaks: two tables in the two columns of one page are two tables."""
    left = make_table(0, [["Name", "Wert"], ["a", "1"]], page_no=1, top=300.0, bottom=60.0,
                      left=50.0, right=290.0)
    right = make_table(1, [["Name", "Wert"], ["b", "2"]], page_no=1, top=730.0, bottom=500.0,
                       left=310.0, right=550.0)

    assert len(paged_constructor.merge_table_continuations([left, right])) == 2


def test_table_ending_high_on_its_page_is_not_a_continuation(paged_constructor: SnippetGraphConstructor):
    """A table that finishes in the upper half of its page was not cut by the page end."""
    head = make_table(0, [["Name", "Wert"], ["a", "1"]], page_no=1, top=730.0, bottom=600.0)
    next_page = make_table(1, [["Name", "Wert"], ["b", "2"]], page_no=2, top=730.0, bottom=400.0)

    assert len(paged_constructor.merge_table_continuations([head, next_page])) == 2


def test_merging_twice_does_not_accumulate_continuation_refs(paged_constructor: SnippetGraphConstructor):
    head = make_table(0, [["Teststreifenbedarf", ">4"], ["Messintervall", "taglich"]],
                      page_no=1, top=730.0, bottom=60.0)
    tail = make_table(1, [["Teststreifenbedarf", ">2"], ["Messintervall", "woechentlich"]],
                      page_no=2, top=730.0, bottom=400.0)

    assert len(paged_constructor.merge_table_continuations([head, tail])) == 1
    assert paged_constructor._table_continuations == {"#/tables/0": ["#/tables/1"]}

    # a second run must not accumulate the refs of the first
    assert len(paged_constructor.merge_table_continuations([head, tail])) == 1
    assert paged_constructor._table_continuations == {"#/tables/0": ["#/tables/1"]}


def test_an_address_closing_on_an_email_is_finished(paged_constructor: SnippetGraphConstructor):
    """The masthead lists one correspondence entry per author, each ending on an
    e-mail address. Ending on a lower case word, it would otherwise read as cut off
    and swallow the next author."""
    first = sidebar_snippet(0, top=250.0, text="Klinik Ostfildern, Deutschland a.zeyfang@medius-kliniken.de")
    second = sidebar_snippet(1, top=238.0, text="PD Dr. med. Anke Bahrmann, Universitätsklinikum Heidelberg")

    assert len(paged_constructor.merge_line_fragments([first, second])) == 2
    assert len(paged_constructor.stitch_continuations([first, second])) == 2


def test_page_geometry_reads_docling_page_sizes_in_page_order(
    paged_constructor: SnippetGraphConstructor,
):
    """The page table comes off the parsed docling document, not off pdf_doc.

    Both report the same numbers, but only the docling document survives in the
    parse cache, so a graph rebuilt from the cache still carries its pages.
    """
    paged_constructor.docling_doc = SimpleNamespace(
        pages={
            2: SimpleNamespace(page_no=2, size=SimpleNamespace(width=595.0, height=842.0)),
            1: SimpleNamespace(page_no=1, size=SimpleNamespace(width=595.0, height=842.0)),
        }
    )
    pages = paged_constructor.page_geometry()
    assert [p.page_no for p in pages] == [1, 2]
    assert [(p.width, p.height) for p in pages] == [(595.0, 842.0), (595.0, 842.0)]


def make_cell(text: str, bottom: float, top: float, left: float = 50.0, right: float = 250.0,
              origin: CoordOrigin = CoordOrigin.BOTTOMLEFT) -> TextCell:
    """One page cell, as a rectangle wound anticlockwise from its bottom-left corner."""
    return TextCell(
        rect=BoundingRectangle(
            r_x0=left, r_y0=bottom, r_x1=right, r_y1=bottom,
            r_x2=right, r_y2=top, r_x3=left, r_y3=top, coord_origin=origin,
        ),
        text=text, orig=text, from_ocr=False,
    )


class CountingPage(SimpleNamespace):
    """A page that records how often its cells were read."""

    def __init__(self, cells: list[TextCell]):
        super().__init__(dimension=SimpleNamespace(width=PAGE_WIDTH, height=PAGE_HEIGHT))
        self._cells = cells
        self.reads = 0

    def iterate_cells(self, cell_unit):
        self.reads += 1
        return list(self._cells)


def make_cell_constructor(page: CountingPage) -> SnippetGraphConstructor:
    c = SnippetGraphConstructor.__new__(SnippetGraphConstructor)
    c._page_cells_cache = OrderedDict()
    c.pdf_doc = SimpleNamespace(get_page=lambda page_no: page)
    return c


def test_cells_in_bbox_returns_only_the_cells_the_box_covers():
    page = CountingPage([make_cell("inside", 100.0, 110.0), make_cell("elsewhere", 600.0, 610.0)])
    c = make_cell_constructor(page)

    box = BoundingBox(l=50.0, t=115.0, r=250.0, b=95.0, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert [cell.text for cell in c.cells_in_bbox(1, TextCellUnit.LINE, box)] == ["inside"]


def test_cells_of_a_page_are_read_once_however_many_snippets_ask():
    """The point of the cache. Reading per snippet made a copy of the whole page
    per snippet, which was almost all of the time graph construction took."""
    page = CountingPage([make_cell(f"line {i}", 100.0 + i * 20, 110.0 + i * 20) for i in range(5)])
    c = make_cell_constructor(page)

    for i in range(5):
        box = BoundingBox(l=50.0, t=115.0 + i * 20, r=250.0, b=95.0 + i * 20,
                          coord_origin=CoordOrigin.BOTTOMLEFT)
        assert len(c.cells_in_bbox(1, TextCellUnit.LINE, box)) == 1
    assert page.reads == 1


def test_cells_are_converted_to_the_origin_of_the_query_box():
    """A cell reported top-left has to be comparable with a bottom-left query box,
    or every overlap test against docling provenance silently misses."""
    page = CountingPage([make_cell("top-left", 683.0, 693.0, origin=CoordOrigin.TOPLEFT)])
    c = make_cell_constructor(page)

    # the same band of the page, counted from the bottom instead
    box = BoundingBox(l=50.0, t=PAGE_HEIGHT - 683.0, r=250.0, b=PAGE_HEIGHT - 693.0,
                      coord_origin=CoordOrigin.BOTTOMLEFT)
    found = c.cells_in_bbox(1, TextCellUnit.LINE, box)
    assert [cell.text for cell in found] == ["top-left"]
    assert found[0].rect.coord_origin == CoordOrigin.BOTTOMLEFT


def test_converting_a_cell_does_not_rewrite_the_page_it_came_from():
    """Cells are handed out shared, so the conversion must copy rather than
    mutate -- otherwise the first query rewrites the page for every later one."""
    original = make_cell("top-left", 683.0, 693.0, origin=CoordOrigin.TOPLEFT)
    c = make_cell_constructor(CountingPage([original]))

    box = BoundingBox(l=50.0, t=PAGE_HEIGHT - 683.0, r=250.0, b=PAGE_HEIGHT - 693.0,
                      coord_origin=CoordOrigin.BOTTOMLEFT)
    c.cells_in_bbox(1, TextCellUnit.LINE, box)
    assert original.rect.coord_origin == CoordOrigin.TOPLEFT
    assert original.rect.r_y0 == 683.0


def test_the_page_cell_cache_stays_bounded():
    """The character cells of a 200-page report are not worth holding at once."""
    page = CountingPage([make_cell("a", 100.0, 110.0)])
    c = make_cell_constructor(page)
    box = BoundingBox(l=50.0, t=115.0, r=250.0, b=95.0, coord_origin=CoordOrigin.BOTTOMLEFT)

    for page_no in range(PAGE_CELL_CACHE_PAGES * 3):
        c.cells_in_bbox(page_no, TextCellUnit.LINE, box)
    assert len(c._page_cells_cache) == PAGE_CELL_CACHE_PAGES
