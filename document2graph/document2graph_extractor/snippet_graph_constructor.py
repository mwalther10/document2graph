from docling_core.types.doc.document import TextItem, PictureItem, TableItem, DoclingDocument # type: ignore
from docling_parse.pdf_parser import PdfDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCellUnit
import copy
import regex as re
import networkx as nx
from collections import Counter, OrderedDict, defaultdict
from ..models.TextSnippet import (
    REGION_BODY,
    REGION_FIGURE,
    REGION_FRONT_MATTER,
    REGION_SIDEBAR,
    TextSnippet,
)
from ..models.TextSnippetNode import TextSnippetNode
from ..models.ImageSnippetNode import ImageSnippetNode
from ..models.TableSnippetNode import TableSnippetNode
from ..models.DocumentMetadata import MetadataExtractionConfig
from ..models.EdgeWeightConfig import EdgeWeightConfig
from ..models.PageGeometry import PageGeometry
from ..models.PipelineFlags import PipelineFlags
from typing import Any, Callable, NamedTuple, TypeVar

from ..edge_weights import (
    RELATION_LIST_ITEM,
    RELATION_MEDIA,
    RELATION_REFERENCE,
    RELATION_ROOT,
    RELATION_SECTION,
    RELATION_TEXT,
    RELATION_UNREFERENCED_MEDIA,
    Edge,
    EdgeContext,
    Unit,
    WeightedEdge,
    apply_edge_weights,
)
from ..edge_weights.context import KIND_IMAGE, KIND_ROOT, KIND_TABLE, KIND_TEXT
from ..utils.log import Log
from ..utils.provenance import provenance_from_prov
from ..utils.tokens import approximate_token_count
from .level_classifier import LevelClassifier
from .metadata_extractor import DocumentMetadataExtractor

T = TypeVar('T', bound=ImageSnippetNode)

ROOT_NODE_ID = "#/document-root"

# joins the labels absorbed into a picture. A chart's ticks and data points were
# never a sentence, and joining them with a space would read as one -- a middot
# keeps them legible as the separate labels they are.
FIGURE_TEXT_SEPARATOR = " · "

# caption recovery: caption-like text ("Tab. 3", "Abb.1", "Figure 2", ...) adjacent
# to a media bbox is treated as its caption. Above/below captions may be up to
# MAX_CAPTION_DISTANCE away; margin captions (rotated, beside the media) must hug
# it within MAX_CAPTION_SIDE_GAP so the search never crosses a column gutter.
# The label is often set behind a typographic marker ("▶ Tab.1 Therapieziele"), which
# belongs to the caption and must not hide it.
CAPTION_PATTERN = re.compile(r"^[\s▶▪●•·‣→>»|–—-]*(Tab|Abb|Table|Figure|Fig)\.?\s*\d+", re.IGNORECASE)
MAX_CAPTION_DISTANCE = 50.0
MAX_CAPTION_SIDE_GAP = 12.0

# decoration filter: pictures shorter than this (pt), or recurring at the same
# position on this many pages, are page decorations (logos, header art)
DECORATION_MAX_HEIGHT = 15.0
DECORATION_MIN_REPEATS = 3

# A thin shape spanning at least RULE_MIN_WIDTH of the page is a horizontal rule.
# Such a rule divides the page into blocks that are read one after the other: in a
# multi-column layout everything above it belongs together, whatever column it sits
# in. Docling reads column by column instead, which interleaves the two-column front
# matter of the first page with the start of the body.
RULE_MIN_WIDTH = 0.6
RULE_MAX_HEIGHT = 3.0

# A shape with area is a box: the guideline layout sets sidebars (recommendation
# boxes, changelogs, infoboxes) in a shaded rectangle. Docling labels their titles
# as section headers, but they are asides, not sections of the document.
BOX_MIN_WIDTH = 40.0
BOX_MIN_HEIGHT = 20.0
# A rectangle covering this much of the page in both directions is its background, not
# a sidebar: word processors lay one under every page. Counting it as a box puts the
# whole page inside a sidebar, which leaves the document without body text and so
# without an outline to rank its headings into.
BOX_MAX_PAGE_COVERAGE = 0.95
# tolerance when testing whether a snippet sits inside a box (pt)
BOX_PADDING = 2.0

# two line cells further apart than this (pt) sit on different text rows
ROW_GAP = 2.0

# Continuation stitching: labels whose text can run on across a column or page break,
# and the shape of a fragment that is cut mid-sentence.
CONTINUATION_LABELS = ("text", "list_item", "footnote")
TERMINAL_PUNCTUATION = re.compile(r'[.!?:;)\]"”’»]\s*$')
CONTINUATION_BREAKS = ",-­"
# a hyphen before one of these words is suspended ("Akut- und Folgekomplikationen"),
# not a word broken across the break
SUSPENDED_HYPHEN_FOLLOWERS = ("und", "oder", "bzw", "sowie", "beziehungsweise", "and", "or")
# Things that legitimately open a column without continuing the previous one, and so
# must never be stitched onto it: a numbered affiliation or reference ("14 Deutsches
# Zentrum ..."), and a bibliography entry, which opens on a surname and initials
# ("Pani LN, Korenda L, Meigs JB et al."). Captions are caught by CAPTION_PATTERN.
NUMBERED_ENTRY = re.compile(r"^\d+[\s.)]")
BIBLIOGRAPHY_ENTRY = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?\s+[A-Z]{1,3}[,.]")
# An address closes on its e-mail, whatever punctuation follows it. Without this a
# correspondence entry counts as unfinished -- it ends on a lower case word -- and runs
# into the next author of the masthead.
TRAILING_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}[.,;:]?$")

# Line-fragment merging. Docling reads a figure's body, a boxed sidebar and the
# masthead line by line instead of block by block, so one bullet of a flowchart
# becomes five snippets. Running text is not affected -- there docling's own
# clustering holds and stitch_continuations covers the column breaks -- so the
# merge is restricted to the regions that are set line by line.
# Calibrated against a hand-labelled review of 298 candidate splits across the
# corpus: within a region, two fragments at most MAX_LINE_FRAGMENT_GAP apart
# were one block in 97% of cases, and none more than 10pt apart ever were.
LINE_FRAGMENT_REGIONS = (REGION_FIGURE, REGION_SIDEBAR, REGION_FRONT_MATTER)
MAX_LINE_FRAGMENT_GAP = 4.0
# tall display type in a figure sets its lines further apart than body text
LINE_FRAGMENT_GAP_RATIO = 0.35
# fraction of the narrower fragment the two must share horizontally to be read as
# one column; keeps the two halves of a side-by-side figure apart
MIN_LINE_FRAGMENT_OVERLAP = 0.3

# Table continuation. A table continued on the next page is two TableItems, and
# nothing downstream rejoins them. The same review found continuations only ever
# across a *page* break: every candidate pair that met across a column break was
# two separate tables (the recommendation boxes of the guideline layout, which
# repeat their header row and so cannot be told apart by it).
TABLE_CONTINUATION_MIN_STUB_OVERLAP = 0.6
# one shared row label is the repeated "Empfehlungen" of two sibling boxes; a real
# continuation repeats its whole stub
TABLE_CONTINUATION_MIN_SHARED_LABELS = 2
TABLE_CONTINUATION_MAX_WIDTH_DELTA = 5.0
# a continuation ends its page low and resumes high on the next
TABLE_CONTINUATION_MAX_BOTTOM = 0.35
TABLE_CONTINUATION_MIN_TOP = 0.6

# How many (page, cell unit, origin) cell lists to keep. Snippets arrive in page
# order, so a handful is enough to hit on nearly every lookup, while holding the
# character cells of a whole 200-page report at once would not be.
PAGE_CELL_CACHE_PAGES = 8

SnippetNode = TextSnippetNode | ImageSnippetNode | TableSnippetNode

class SnippetGraph(NamedTuple):
    """Result of get_graph: all snippet nodes, weighted edges, and the id of the root node.

    edges form the hierarchy tree (one parent per node); reference_edges link every
    text snippet that mentions a media item to that item and are not part of the tree.

    root_id is the snippet_id of the document title node, or the synthetic
    ROOT_NODE_ID (which is not part of the node lists) if no title node was found.

    pages carries the size of every page of the source document, so that the
    boxes on the nodes can be placed on a page by a consumer that no longer has
    the PDF.

    relations is the structural relation of every edge, keyed by (parent, child).
    Three of them -- reference, unreferenced_media and root -- are facts about how
    the edge was created rather than about the two nodes, so they cannot be
    recovered from the nodes afterwards and have to travel with the graph.

    Both are last and default to empty because the tuple is built positionally in
    places that predate them.
    """
    text_nodes: list[TextSnippetNode]
    image_nodes: list[ImageSnippetNode]
    table_nodes: list[TableSnippetNode]
    edges: list[WeightedEdge]
    reference_edges: list[WeightedEdge]
    root_id: str
    pages: list[PageGeometry] = []
    relations: dict[Edge, str] = {}

class SnippetGraphConstructor():
    def __init__(self, pdf_doc: PdfDocument, docling_doc: DoclingDocument, filename: str, document_type: str, metadata_config: MetadataExtractionConfig | None = None, edge_weights: EdgeWeightConfig | None = None, flags: PipelineFlags | None = None):
        self.pdf_doc = pdf_doc
        self.docling_doc = docling_doc
        self.edge_weights = edge_weights or EdgeWeightConfig()
        self._edge_relations: dict[Edge, str] = {}
        self.metadata_config = metadata_config or MetadataExtractionConfig()
        self.flags = flags or PipelineFlags()
        self.logger = Log("SnippetGraphConstructor").logger
        self._page_shapes_cache: dict[int, tuple[list[float], list[tuple[float, float, float, float]]]] = {}
        self._page_cells_cache: OrderedDict = OrderedDict()
        self.text_items = self._prepare_text_items(docling_doc)
        # head table ref -> refs of the continuations folded into it; merge_table_continuations
        # fills it, so it has to exist even when the step is off
        self._table_continuations: dict[str, list[str]] = {}
        self.table_items = (self.merge_table_continuations(docling_doc.tables)
                            if self.flags.merge_table_continuations else list(docling_doc.tables))
        self.image_items = self.filter_decorative_pictures(docling_doc.pictures) if self.flags.filter_decorative_pictures else list(docling_doc.pictures)
        self._document_metadata = DocumentMetadataExtractor(self.text_items).extract(filename, document_type, self.metadata_config)
        self.levels = LevelClassifier(self.text_items, self._document_metadata.title, level_source=self.flags.level_source)

    def _prepare_text_items(self, docling_doc: DoclingDocument) -> list[TextSnippet]:
        """Read the text snippets off the parsed document and run the repair steps
        the flags leave enabled, in order: reading order, then stitching (which
        needs the repaired order), then regions."""
        snippets = [TextSnippet(text_item=item, line_heights=self.add_line_heights(item), font_key=self.get_font_key(item))
                    for item in docling_doc.texts if item.label not in ("page_footer", "page_header")]
        if self.flags.repair_reading_order:
            snippets = self.order_by_page_blocks(snippets)
        if self.flags.stitch_continuations:
            snippets = self.stitch_continuations(snippets)
        if self.flags.assign_regions:
            snippets = self.assign_regions(snippets)
            if self.flags.merge_line_fragments:
                # last, because it reads the region the previous step assigned
                snippets = self.merge_line_fragments(snippets)
        return snippets

    @property
    def document_metadata(self):
        return self._document_metadata

    def page_geometry(self) -> list[PageGeometry]:
        """The box of every page of the document, in page order.

        Read off the parsed docling document rather than off ``pdf_doc``: the
        boxes stored on the nodes are docling provenance boxes, so docling's own
        page sizes are the space they are already expressed in. The two agree
        anyway -- both report the PDF's points -- but only one of them survives
        in the cached parse.
        """
        return [
            PageGeometry(page_no=page_no, width=page.size.width, height=page.size.height)
            for page_no, page in sorted(self.docling_doc.pages.items())
        ]

    def _page_cells(self, page_no: int, cell_unit: TextCellUnit, origin: CoordOrigin):
        """Every cell of one page, converted to ``origin`` once, with its bounding box.

        ``SegmentedPdfPage.get_cells_in_bbox`` deep-copies *every* cell on the
        page before testing a single one for overlap, so asking it for the cells
        of one snippet costs a copy of the whole page. Graph construction asks
        once per snippet, twice over (line heights and font keys), which makes
        the total (snippets x cells per page) copies of a pydantic model: 90 of
        the 95 seconds a 12-page guideline took to build, almost all of it under
        get_font_key, whose character cells are the numerous kind.

        The cells of a page do not change between those calls, so they are read
        and converted once here and the overlap test is done against the cached
        boxes. Only a few pages are held at a time: snippets arrive in page
        order, so a small cache hits nearly always, and the character cells of a
        200-page report would not be worth keeping all at once.
        """
        key = (page_no, cell_unit, origin)
        cached = self._page_cells_cache.get(key)
        if cached is not None:
            self._page_cells_cache.move_to_end(key)
            return cached

        page = self.pdf_doc.get_page(page_no)
        entries = []
        for cell in page.iterate_cells(cell_unit):
            # copy only when the rect has to be rewritten, and never per query
            if cell.rect.coord_origin != origin:
                cell = copy.deepcopy(cell)
                cell.rect = (cell.rect.to_top_left_origin(page.dimension.height)
                             if origin == CoordOrigin.TOPLEFT
                             else cell.rect.to_bottom_left_origin(page.dimension.height))
            entries.append((cell, cell.to_bounding_box()))

        self._page_cells_cache[key] = entries
        while len(self._page_cells_cache) > PAGE_CELL_CACHE_PAGES:
            self._page_cells_cache.popitem(last=False)
        return entries

    def cells_in_bbox(self, page_no: int, cell_unit: TextCellUnit, bbox: BoundingBox,
                      ios: float = 0.8) -> list:
        """The cells of a page that ``bbox`` covers.

        The semantics of ``SegmentedPdfPage.get_cells_in_bbox`` -- same origin
        handling, same intersection-over-self threshold -- without its per-call
        copy of the page.

        The cells come back shared rather than copied, so a caller must treat
        them as read-only: writing to one would rewrite what the next query on
        the same page sees. Every caller here only measures them.
        """
        return [cell for cell, cell_bbox in self._page_cells(page_no, cell_unit, bbox.coord_origin)
                if cell_bbox.intersection_over_self(bbox) > ios]

    def add_line_heights(self, snippet: TextItem) -> list[float]:
        snippet_bbox = snippet.prov[0].bbox
        lines = self.cells_in_bbox(snippet.prov[0].page_no, TextCellUnit.LINE, snippet_bbox)
        heights = []
        for line in lines:
            # check if line is in snippet bbox
            line_bbox = line.rect
            if line.text in snippet.text:
                heights.append(
                        ((line_bbox.r_y3 - line_bbox.r_y0) + (line_bbox.r_y2 - line_bbox.r_y1)) / 2
                    )
        # Some embedded display/all-caps heading fonts report a degenerate ~1-2pt
        # glyph box for their line (and char) cells, collapsing a large heading to a
        # tiny height and inverting its level. Detect this by comparing the measured
        # height to the geometric line pitch (snippet bbox height / number of rows):
        # when the glyph box is far below the pitch, fall back to the pitch, which
        # tracks the real font size. Well-behaved fonts (height ~= pitch) and ordinary
        # body text are left untouched.
        rows = self.count_text_rows(lines)
        if heights and rows:
            line_pitch = abs(snippet_bbox.t - snippet_bbox.b) / rows
            if max(heights) < 0.5 * line_pitch:
                heights = [line_pitch] * len(heights)
        return [round(h, 1) for h in heights]

    @staticmethod
    def count_text_rows(lines) -> int:
        """Number of text rows the line cells form.

        For the same fonts that report a degenerate glyph box, docling_parse also splits
        a line into one cell per word. Counting cells would then divide the snippet
        height by the word count and produce a pitch as tiny as the broken glyph box
        itself, so the rows are recovered from the vertical positions instead."""
        centers = sorted((cell.rect.r_y0 + cell.rect.r_y3) / 2 for cell in lines)
        rows = 1 if centers else 0
        for lower, upper in zip(centers, centers[1:]):
            if upper - lower > ROW_GAP:
                rows += 1
        return rows

    def _page_shapes(self, page_no: int) -> tuple[list[float], list[tuple[float, float, float, float]]]:
        """The drawn shapes of a page that carry structure, in bottom-left origin:
        the page-spanning horizontal rules (as y positions, descending) and the boxes
        (as l, r, b, t). Everything else -- hairlines, bullets, logos -- is ignored."""
        if page_no not in self._page_shapes_cache:
            page = self.pdf_doc.get_page(page_no)
            width, height = page.dimension.width, page.dimension.height
            dividers, boxes = [], []
            for shape in page.shapes:
                xs = [point[0] for point in shape.points]
                ys = [point[1] for point in shape.points]
                if shape.coord_origin == CoordOrigin.TOPLEFT:
                    ys = [height - y for y in ys]
                left, right, bottom, top = min(xs), max(xs), min(ys), max(ys)
                if top - bottom <= RULE_MAX_HEIGHT and right - left >= RULE_MIN_WIDTH * width:
                    dividers.append(top)
                covers_page = (right - left >= BOX_MAX_PAGE_COVERAGE * width
                               and top - bottom >= BOX_MAX_PAGE_COVERAGE * height)
                if not covers_page and right - left >= BOX_MIN_WIDTH and top - bottom >= BOX_MIN_HEIGHT:
                    boxes.append((left, right, bottom, top))
            self._page_shapes_cache[page_no] = (sorted(set(dividers), reverse=True), boxes)
        return self._page_shapes_cache[page_no]

    def page_dividers(self, page_no: int) -> list[float]:
        """Descending y positions (bottom-left origin) of the horizontal rules that span
        the page, i.e. the boundaries between the reading blocks of that page."""
        return self._page_shapes(page_no)[0]

    def page_boxes(self, page_no: int) -> list[tuple[float, float, float, float]]:
        """(l, r, b, t) of the drawn boxes of a page, bottom-left origin."""
        return self._page_shapes(page_no)[1]

    def in_box(self, snippet: TextSnippet) -> bool:
        """Whether a snippet sits inside a drawn box, i.e. in a sidebar."""
        bbox = self._normalized_bbox(snippet)
        return any(left - BOX_PADDING <= bbox.l and bbox.r <= right + BOX_PADDING
                   and bottom - BOX_PADDING <= bbox.b and bbox.t <= top + BOX_PADDING
                   for left, right, bottom, top in self.page_boxes(snippet.text_item.prov[0].page_no))

    def block_of(self, snippet: TextSnippet) -> tuple[int, int]:
        """(page, block) of a snippet. Blocks are numbered from the top of the page and
        change at every page-spanning rule."""
        prov = snippet.text_item.prov[0]
        page_height = self.pdf_doc.get_page(prov.page_no).dimension.height
        top = prov.bbox.to_bottom_left_origin(page_height).t
        return prov.page_no, sum(1 for divider in self.page_dividers(prov.page_no) if top < divider)

    def order_by_page_blocks(self, snippets: list[TextSnippet]) -> list[TextSnippet]:
        """Reading order with the page blocks restored: snippets are sorted by the block
        they sit in, keeping docling's order within a block.

        Docling emits a two-column page column by column, so a block that is continued in
        the right column (the front matter of the first page) is split apart by whatever
        the left column already holds of the next block. Everything downstream -- sequence
        numbers, parent lookup, grouping -- reads this order, so it is repaired here."""
        ordered = sorted(enumerate(snippets), key=lambda item: (self.block_of(item[1]), item[0]))
        return [snippet for _, snippet in ordered]

    def reading_key(self, page_no: int, bbox: BoundingBox) -> tuple[int, int, int, float]:
        """Where an item sits in the reading order of the document, as a sortable key.

        Text nodes are numbered in reading order, but media nodes are numbered within
        the media list, so the two cannot be compared by sequence_no; this key places
        them against each other. It follows the same layout rules as the node order:
        blocks are read one after the other (see order_by_page_blocks), and within a
        block the left column precedes the right one."""
        page = self.pdf_doc.get_page(page_no)
        box = bbox.to_bottom_left_origin(page.dimension.height)
        block = sum(1 for divider in self.page_dividers(page_no) if box.t < divider)
        column = 0 if (box.l + box.r) / 2 < page.dimension.width / 2 else 1
        return page_no, block, column, -box.t

    def column_of(self, snippet: TextSnippet) -> int:
        """Which half of the page a snippet sits in. The corpus is set in two columns;
        a page with more columns is only split into left and right here."""
        prov = snippet.text_item.prov[0]
        page = self.pdf_doc.get_page(prov.page_no)
        bbox = prov.bbox.to_bottom_left_origin(page.dimension.height)
        return 0 if (bbox.l + bbox.r) / 2 < page.dimension.width / 2 else 1

    def stitch_continuations(self, snippets: list[TextSnippet]) -> list[TextSnippet]:
        """Join paragraphs that docling cut at a column or page break.

        A paragraph running from the bottom of one column into the top of the next is two
        layout clusters, so docling emits two text items and everything downstream sees
        two snippets split mid-sentence. Docling repairs some of these itself, but only
        from one TEXT item into another; the split typically starts in a list item and
        continues in a text item, because the continuation carries no bullet and is
        labelled accordingly.

        Two snippets are joined when the first ends its column unfinished and the second
        opens the next column (or the next page) resuming in lower case. Being adjacent in
        reading order is not enough: both must sit at the ends of their columns, which is
        what keeps two-column glossaries and term/definition pairs apart. The joined
        snippet keeps the geometry of its opening fragment and records the continuation's
        provenance."""
        flowing = [s for s in snippets
                   if s.text_item.label in CONTINUATION_LABELS and not self._inside_picture(s)]
        column_bottom: dict[tuple[int, int, int], float] = {}
        column_top: dict[tuple[int, int, int], float] = {}
        for snippet in flowing:
            key = (*self.block_of(snippet), self.column_of(snippet))
            bbox = self._normalized_bbox(snippet)
            column_bottom[key] = min(column_bottom.get(key, bbox.b), bbox.b)
            column_top[key] = max(column_top.get(key, bbox.t), bbox.t)

        stitched: list[TextSnippet] = []
        tail: TextSnippet | None = None  # last fragment absorbed into stitched[-1]
        for snippet in snippets:
            previous = tail or (stitched[-1] if stitched else None)
            if previous is not None and self._continues(previous, snippet, column_bottom, column_top):
                self.logger.info(f"Stitching {snippet.text_item.self_ref} onto {stitched[-1].text_item.self_ref} across a column break.")
                stitched[-1] = self._merge_continuation(stitched[-1], snippet)
                tail = snippet
            else:
                stitched.append(snippet)
                tail = None
        return stitched

    def _continues(self, previous: TextSnippet, snippet: TextSnippet,
                   column_bottom: dict[tuple[int, int, int], float],
                   column_top: dict[tuple[int, int, int], float]) -> bool:
        """Whether `snippet` picks up the sentence `previous` breaks off at its column end."""
        if not self._is_unfinished(previous.text_item.text) or not self._resumes(snippet.text_item.text):
            return False
        previous_key = (*self.block_of(previous), self.column_of(previous))
        snippet_key = (*self.block_of(snippet), self.column_of(snippet))
        if previous_key not in column_bottom or snippet_key not in column_top:
            return False  # a heading, caption or figure label: not part of the text flow
        (previous_page, previous_block, previous_column) = previous_key
        (snippet_page, snippet_block, snippet_column) = snippet_key
        if snippet_page == previous_page:
            # the text flows on into a column further right of the same block
            if snippet_block != previous_block or snippet_column <= previous_column:
                return False
        elif snippet_page < previous_page:
            return False
        # the break must be the end of one column and the start of the next, not just two
        # neighbours in reading order
        return (self._normalized_bbox(previous).b == column_bottom[previous_key]
                and self._normalized_bbox(snippet).t == column_top[snippet_key])

    def _merge_continuation(self, head: TextSnippet, tail: TextSnippet) -> TextSnippet:
        """Fold a continuation fragment into the snippet it belongs to, keeping the
        opening fragment's geometry as the snippet's position."""
        text = self._join_continuation(head.text_item.text, tail.text_item.text)
        orig = self._join_continuation(head.text_item.orig, tail.text_item.orig)
        text_item = head.text_item.model_copy(update={
            "text": text, "orig": orig, "prov": [*head.text_item.prov, *tail.text_item.prov],
        })
        return head.model_copy(update={
            "text_item": text_item,
            "line_heights": [*head.line_heights, *tail.line_heights],
        })

    @staticmethod
    def _join_continuation(head: str, tail: str) -> str:
        head, tail = head.rstrip(), tail.lstrip()
        if head.endswith("­"):  # soft hyphen: always a word broken across lines
            return head[:-1] + tail
        if head.endswith("-"):
            first_word = re.split(r"\W+", tail, maxsplit=1)[0].lower()
            if first_word not in SUSPENDED_HYPHEN_FOLLOWERS:
                return head[:-1] + tail
        return f"{head} {tail}"

    @staticmethod
    def _inside_picture(snippet: TextSnippet) -> bool:
        """Text that docling read out of a figure: it is not part of the running text."""
        parent = snippet.text_item.parent
        return parent is not None and "pictures" in parent.get_ref().cref

    @staticmethod
    def _is_unfinished(text: str) -> bool:
        """Whether a fragment breaks off mid-sentence: no terminal punctuation, and ending
        on a comma, a hyphen, or a lower case *word*.

        The last word decides it, not the last character: "Universitätsmedizin
        Greifswald, Deutschland" ends on a lower case letter but is a complete
        affiliation, while "... Nierenersatztherapie, koronare" ends on a lower case
        word that demands the noun following it across the break."""
        stripped = text.rstrip()
        if not stripped or TERMINAL_PUNCTUATION.search(text):
            return False
        if TRAILING_EMAIL.search(stripped):
            return False
        if stripped[-1] in CONTINUATION_BREAKS:
            return True
        return stripped.split()[-1][:1].islower()

    @staticmethod
    def _resumes(text: str) -> bool:
        """Whether a fragment picks up the sentence the previous one broke off.

        A lower case opening is the clearest case, but German resumes on a capitalised
        noun about as often ("... koronare" / "Herzkrankheit, periphere arterielle
        ..."), so case alone cannot decide this. What the case test was really keeping
        out is the short list of things that legitimately open a column without
        continuing anything: a caption, a numbered affiliation or reference, a new
        bibliography entry. Those are excluded by name instead, which is what lets a
        capitalised continuation through."""
        stripped = text.lstrip()
        if not stripped or CAPTION_PATTERN.match(stripped):
            return False
        # an opening bracket belongs to the continuation: "... Qualitätskontrolle" /
        # "(Kontrolllösungsmessung) erfüllen, ..."
        stripped = stripped.lstrip("([")
        if not stripped or NUMBERED_ENTRY.match(stripped) or BIBLIOGRAPHY_ENTRY.match(stripped):
            return False
        return stripped[0].isalpha()

    def merge_line_fragments(self, snippets: list[TextSnippet]) -> list[TextSnippet]:
        """Join the consecutive lines of one text block inside a figure, a sidebar or
        the masthead.

        Docling clusters running text into paragraphs, but reads these three regions
        line by line: a five-line bullet of a flowchart arrives as five snippets, each
        a sentence fragment, and every one of them becomes its own graph node. They are
        rejoined here on geometry -- same column, directly underneath, horizontally
        overlapping -- combined with the same continuation tests
        :meth:`stitch_continuations` uses, so only fragments that read as one running
        text are joined and two separate labels of a diagram are left alone.

        Body text never enters this: there docling's clustering is right, and joining
        neighbouring bullets of an enumeration would destroy the list."""
        merged: list[TextSnippet] = []
        tail: TextSnippet | None = None  # last fragment folded into merged[-1]
        for snippet in snippets:
            previous = tail or (merged[-1] if merged else None)
            if previous is not None and self._is_line_fragment_of(previous, snippet):
                self.logger.debug(
                    f"Merging line fragment {snippet.text_item.self_ref} into "
                    f"{merged[-1].text_item.self_ref} ({snippet.region}).")
                merged[-1] = self._merge_continuation(merged[-1], snippet)
                tail = snippet
            else:
                merged.append(snippet)
                tail = None
        if len(merged) < len(snippets):
            self.logger.info(f"Merged {len(snippets) - len(merged)} line fragments into "
                             f"{len(merged)} snippets (was {len(snippets)}).")
        return merged

    def _is_line_fragment_of(self, previous: TextSnippet, snippet: TextSnippet) -> bool:
        """Whether `snippet` is the next line of the block `previous` ends."""
        if previous.region != snippet.region or snippet.region not in LINE_FRAGMENT_REGIONS:
            return False
        if previous.text_item.prov[0].page_no != snippet.text_item.prov[0].page_no:
            return False
        if not self._is_unfinished(previous.text_item.text) or not self._resumes(snippet.text_item.text):
            return False
        # the previous snippet may already hold several merged lines: measure the gap
        # from the last of them, which is the prov this one follows
        above = previous.text_item.prov[-1].bbox.to_bottom_left_origin(
            self.pdf_doc.get_page(previous.text_item.prov[-1].page_no).dimension.height)
        below = self._normalized_bbox(snippet)
        gap = above.b - below.t
        if gap < 0 or gap > max(MAX_LINE_FRAGMENT_GAP,
                                LINE_FRAGMENT_GAP_RATIO * min(above.t - above.b, below.t - below.b)):
            return False
        overlap = min(above.r, below.r) - max(above.l, below.l)
        narrower = min(above.r - above.l, below.r - below.l)
        return narrower > 0 and overlap / narrower > MIN_LINE_FRAGMENT_OVERLAP

    def _normalized_bbox(self, snippet: TextSnippet):
        prov = snippet.text_item.prov[0]
        return prov.bbox.to_bottom_left_origin(self.pdf_doc.get_page(prov.page_no).dimension.height)

    def find_front_matter(self, snippets: list[TextSnippet]) -> set[str]:
        """Ids of the snippets that make up the front matter: the block above the lowest
        rule of the title page, which holds the masthead (authors, institutes,
        bibliography, correspondence) in a column layout of its own.

        Empty when that page has no such block -- fewer than two rules, or nothing below
        the lowest one, in which case the rule does not separate a front matter."""
        page = self.metadata_config.title_page
        dividers = self.page_dividers(page)
        if len(dividers) < 2:
            return set()
        blocks = {snippet.text_item.self_ref: self.block_of(snippet) for snippet in snippets}
        front_matter, body = (page, len(dividers) - 1), (page, len(dividers))
        if body not in blocks.values():
            return set()
        return {ref for ref, block in blocks.items() if block == front_matter}

    def assign_regions(self, snippets: list[TextSnippet]) -> list[TextSnippet]:
        """Tag every snippet with the part of the page it belongs to.

        Only body text forms the document outline. Docling labels the title of a
        recommendation box, the caption inside a figure and the masthead of the first
        page as section headers just like a real section heading, and a single one of
        them at the wrong level drags the whole hierarchy with it -- a two-page infobox
        looks exactly like a chapter that contains every section between its title and
        its continuation. Telling them apart is a question of where they sit on the
        page, not of how they are set."""
        front_matter = self.find_front_matter(snippets)
        regions = []
        for snippet in snippets:
            if self._inside_picture(snippet):
                region = REGION_FIGURE
            elif self.in_box(snippet):
                region = REGION_SIDEBAR
            elif snippet.text_item.self_ref in front_matter:
                region = REGION_FRONT_MATTER
            else:
                region = REGION_BODY
            regions.append(snippet.model_copy(update={"region": region}))
        counts = Counter(s.region for s in regions)
        self.logger.info(f"Page regions: {dict(counts)}")
        return regions

    def get_font_key(self, snippet: TextItem) -> str | None:
        """Dominant embedded-font key of a snippet, used to rank section-header
        levels by font identity. Font size (line height) is an unreliable ranking
        signal on its own -- some documents set subsections in a taller face than
        the sections that contain them -- whereas the font key is stable per style.
        Returns None when no font information is available."""
        chars = self.cells_in_bbox(snippet.prov[0].page_no, TextCellUnit.CHAR, snippet.prov[0].bbox)
        keys = Counter(cell.font_key for cell in chars if getattr(cell, "font_key", None))
        return keys.most_common(1)[0][0] if keys else None

    def filter_decorative_pictures(self, pictures: list[PictureItem]) -> list[PictureItem]:
        """Drop repeated page decorations (logos, header/footer art) that docling
        leaves in the body layer: tiny pictures, or pictures recurring at the
        same position on several pages. They carry no retrievable content and
        can never be caption-matched."""
        pages_by_position = defaultdict(set)
        for pic in pictures:
            pages_by_position[self._bbox_position_key(pic)].add(pic.prov[0].page_no)
        kept = []
        for pic in pictures:
            bbox = pic.prov[0].bbox
            is_tiny = abs(bbox.t - bbox.b) < DECORATION_MAX_HEIGHT
            is_repeated = len(pages_by_position[self._bbox_position_key(pic)]) >= DECORATION_MIN_REPEATS
            if is_tiny or is_repeated:
                self.logger.info(f"Dropping decorative picture {pic.self_ref} on page {pic.prov[0].page_no} (tiny={is_tiny}, repeated={is_repeated}).")
            else:
                kept.append(pic)
        return kept

    def merge_table_continuations(self, tables: list[TableItem]) -> list[TableItem]:
        """Join a table that runs on to the next page back into one table item.

        ``docling_doc.tables`` holds one item per layout cluster, and a table broken
        by a page break is two of them: two table nodes with half the rows each, the
        caption on the first only, and a continuation that no caption match can ever
        reach. The rows of the second are appended to the first, which keeps the
        node's markdown and HTML whole and lets ``provenance`` report both pages.

        A continuation is recognised by its stub -- the row labels a continued table
        repeats -- or, when the halves share no labels, by resuming at the same width
        without a header row of its own. Both also have to sit at the page break: low
        on one page, high on the next."""
        self._table_continuations = {}
        merged: list[TableItem] = []
        for table in tables:
            head = merged[-1] if merged else None
            if head is not None and self._continues_table(head, table):
                self.logger.info(f"Merging table {table.self_ref} into {head.self_ref}: "
                                 f"continued from page {head.prov[-1].page_no} to {table.prov[0].page_no}.")
                self._table_continuations.setdefault(head.self_ref, []).append(table.self_ref)
                merged[-1] = self._merge_table(head, table)
            else:
                merged.append(table)
        return merged

    def _continues_table(self, head: TableItem, tail: TableItem) -> bool:
        """Whether `tail` is the rest of the table `head` breaks off at the page end."""
        if not head.prov or not tail.prov:
            return False
        head_prov, tail_prov = head.prov[-1], tail.prov[0]
        if tail_prov.page_no != head_prov.page_no + 1:
            return False  # only page breaks: a column break separates two tables here
        head_box = head_prov.bbox.to_bottom_left_origin(self.pdf_doc.get_page(head_prov.page_no).dimension.height)
        tail_box = tail_prov.bbox.to_bottom_left_origin(self.pdf_doc.get_page(tail_prov.page_no).dimension.height)
        if head_box.b > TABLE_CONTINUATION_MAX_BOTTOM * self.pdf_doc.get_page(head_prov.page_no).dimension.height:
            return False
        if tail_box.t < TABLE_CONTINUATION_MIN_TOP * self.pdf_doc.get_page(tail_prov.page_no).dimension.height:
            return False
        if self._repeats_stub(head, tail):
            return True
        # no shared row labels: a continuation then has to resume in the same shape,
        # and carry no header row of its own
        same_shape = (head.data.num_cols == tail.data.num_cols
                      and abs((head_box.r - head_box.l) - (tail_box.r - tail_box.l)) <= TABLE_CONTINUATION_MAX_WIDTH_DELTA)
        return same_shape and not self._has_header_row(tail)

    @staticmethod
    def _stub(table: TableItem) -> list[str]:
        """The table's row labels: the text of its first column, one entry per row."""
        return [" ".join(row[0].text.split()) for row in table.data.grid if row]

    @classmethod
    def _repeats_stub(cls, head: TableItem, tail: TableItem) -> bool:
        """Whether the two halves carry the same row labels, as a continued table does.

        Two shared labels at least: the guideline layout sets every recommendation in
        its own two-row table headed "Empfehlungen", so a single shared label says
        nothing about whether they belong together."""
        head_labels = {label for label in cls._stub(head) if label}
        tail_labels = {label for label in cls._stub(tail) if label}
        if not head_labels or not tail_labels:
            return False
        shared = head_labels & tail_labels
        return (len(shared) >= TABLE_CONTINUATION_MIN_SHARED_LABELS
                and len(shared) / min(len(head_labels), len(tail_labels)) >= TABLE_CONTINUATION_MIN_STUB_OVERLAP)

    @staticmethod
    def _has_header_row(table: TableItem) -> bool:
        grid = table.data.grid
        return bool(grid) and any(cell.column_header for cell in grid[0])

    @staticmethod
    def _merge_table(head: TableItem, tail: TableItem) -> TableItem:
        """Append the rows of a continuation to the table it continues.

        Row offsets of the continuation are shifted past the head's rows and its
        header flags dropped -- a repeated header is the same header, not a second
        one -- so the merged grid reads as one table. Column counts may differ when
        the two halves were parsed separately; each cell keeps its own column
        offsets and the grid is as wide as the wider half."""
        shift = head.data.num_rows
        continued = [cell.model_copy(update={
            "start_row_offset_idx": cell.start_row_offset_idx + shift,
            "end_row_offset_idx": cell.end_row_offset_idx + shift,
            "column_header": False,
        }) for cell in tail.data.table_cells]
        data = head.data.model_copy(update={
            "table_cells": [*head.data.table_cells, *continued],
            "num_rows": head.data.num_rows + tail.data.num_rows,
            "num_cols": max(head.data.num_cols, tail.data.num_cols),
        })
        return head.model_copy(update={
            "data": data,
            "prov": [*head.prov, *tail.prov],
            # the continuation is rarely captioned; keep whichever caption exists
            "captions": list(head.captions) or list(tail.captions),
        })

    @staticmethod
    def _bbox_position_key(pic: PictureItem) -> tuple[int, int, int, int]:
        # ~5pt tolerance for position jitter of the same decoration between pages
        bbox = pic.prov[0].bbox
        return (round(bbox.l / 5), round(bbox.t / 5), round(bbox.r / 5), round(bbox.b / 5))

    @staticmethod
    def _may_parent(candidate: TextSnippetNode, node: TextSnippetNode) -> bool:
        """Whether `candidate` may hold `node` in the tree.

        Regions do not nest into each other: the masthead is not a chapter of the body,
        and the title of a recommendation box never holds the sections that follow it.
        An aside is the one exception -- a sidebar or figure hangs off the body section
        it is printed in."""
        if candidate.region == node.region:
            return True
        return node.region in (REGION_SIDEBAR, REGION_FIGURE) and candidate.region == REGION_BODY

    def compute_parent_id(self, node: TextSnippetNode, nodes: list[TextSnippetNode]) -> str | None:
        if(node.is_grouped):
            # first case: if node is grouped, parent is the adjacent non-grouped node.
            # A list continued across a column break can end up in one docling group with
            # the list that follows it in the raw reading order, mixing regions; only the
            # part in this node's own region decides where it attaches.
            group = [n for n in nodes
                     if n.docling_parent_ref.get_ref().cref == node.docling_parent_ref.get_ref().cref
                     and n.region == node.region]
            first_bullet = min(group, key=lambda n: n.sequence_no)
            introduction = next((n for n in reversed(nodes[:first_bullet.sequence_no])
                                 if self._may_parent(n, node)), None)
            if introduction is not None:
                return introduction.snippet_id
            # a list that opens its region has nothing to hang off: fall through

        elif(node.docling_parent_ref.get_ref().cref != "#/body"):
            # second case: if node has a meaningful parent_id from docling, use it
            parent_ref = node.docling_parent_ref.get_ref().cref
            return parent_ref
        # default case for text items with inaccurate parent_id from docling: find the
        # closest preceding node with a lower level that may hold this one.
        for prev_node in reversed(nodes[:node.sequence_no]):
            if prev_node.level < node.level and self._may_parent(prev_node, node):
                return prev_node.snippet_id
        # if that is not possible, connect to the document title node (level 0) if it exists
        return next((n.snippet_id for n in nodes if n.level == 0 and n.snippet_id != node.snippet_id), None)

    def compute_text_nodes(self, snippets: list[TextSnippet]) -> list[TextSnippetNode]:
        nodes = []
        for i, snippet in enumerate(snippets):
            if snippet.text_item.label in ("page_footer", "page_header"):
                self.logger.debug(f"Skipping text snippet {snippet.text_item.self_ref} with label {snippet.text_item.label}.")
                continue
            snippet_level, snippet_level_label = self.levels.classify(snippet)
            assert snippet.text_item.parent is not None, f"Text item {snippet.text_item.text} has no parent reference."
            node = TextSnippetNode(
                snippet_id=snippet.text_item.self_ref,
                document_id=self._document_metadata.document_id,
                label=snippet.text_item.label,
                sequence_no=i,
                level=snippet_level,
                level_label=snippet_level_label,
                parent_id=None,  # resolved below once all nodes exist
                docling_parent_ref=snippet.text_item.parent.get_ref(),
                docling_self_ref=snippet.text_item.get_ref(),
                is_grouped="group" in snippet.text_item.parent.get_ref().cref,
                text=snippet.text_item.text,
                bbox=snippet.text_item.prov[0].bbox,
                charspan=snippet.text_item.prov[0].charspan,
                page_no=snippet.text_item.prov[0].page_no,
                # stitch_continuations concatenates the provs of the fragments it
                # merged, so a paragraph broken across columns reports both halves
                provenance=provenance_from_prov(snippet.text_item.prov),
                line_heights=snippet.line_heights,
                font_key=snippet.font_key,
                level_height=self.levels.level_height(snippet_level),
                region=snippet.region,
            )
            nodes.append(node)
        nodes = [node.model_copy(update={"parent_id": self.compute_parent_id(node, nodes)}) for node in nodes]
        for node in nodes:
            if node.parent_id is None:
                self.logger.debug(f"Node {node.text} has no parent_id assigned, it will attach to the document root.")
        return nodes

    def rb_image_parent_matching(self, images: list[T], text_nodes: list[TextSnippetNode]) -> list[T]:
        # find text node that contains reference to image or table using a rule-based approach
        ret_images = images.copy()
        for i, image in enumerate(ret_images):
            matched=False
            if image.caption_nodes is not None:
                # get the full caption text
                caption = " ".join([node.text for node in image.caption_nodes])
                caption_node_ids = [node.snippet_id for node in image.caption_nodes]
                rules = ["Figure", "Fig.", "Table", "Abbildung", "Tabelle", "Abb.", "Tab.", "Abb", "Tab", "Abb .", "Tab .", "Abb ", "Tab "]
                self.logger.debug(f"Image {image.docling_self_ref} caption: {caption}")
                for rule in rules:
                    if rule in caption:
                        # normalize the rule into a tolerant pattern (optional dot, flexible
                        # spacing); \b and (?!\d) keep it from matching inside words or
                        # matching "Tab.1" inside "Tab.10"
                        base = re.escape(rule.strip().rstrip(".").strip())
                        match = re.search(rf"\b{base}\s*\.?\s*(\d+)", caption, re.IGNORECASE)
                        if match:
                            mention_pattern = rf"\b{base}\s*\.?\s*{match.group(1)}(?!\d)"
                            # make sure we do not match caption nodes
                            mentions = [node for node in text_nodes if re.search(mention_pattern, node.text, re.IGNORECASE) and node.snippet_id not in caption_node_ids]
                            if mentions and mentions[0].parent_id is not None:
                                # the first mention determines parent and level (hierarchy tree);
                                # all mentions are kept for reference edges
                                self.logger.info(f"Image {image.snippet_id} matched to text node {mentions[0].snippet_id} using rule '{rule}' with pattern '{mention_pattern}' ({len(mentions)} mention(s)).")
                                ret_images[i] = image.model_copy(update={
                                    "level": mentions[0].level,
                                    "level_label": mentions[0].level_label,
                                    "parent_id": mentions[0].parent_id,
                                    "referencing_node_ids": [m.snippet_id for m in mentions],
                                })
                                matched=True
                                break
            if not matched:
                # no caption or no match: assign to the section it is printed in, i.e. the
                # closest heading above it in reading order. Asides are skipped: a figure
                # label or the title of a sidebar box heads no section of the document, so
                # a table printed past one still belongs to the body section around it.
                image_key = self.reading_key(image.page_no, image.bbox)
                preceding_headers = [node for node in text_nodes
                                     if node.level in self.levels.header_levels()
                                     and node.region not in (REGION_SIDEBAR, REGION_FIGURE)
                                     and self.reading_key(node.page_no, node.bbox) < image_key]
                if preceding_headers:
                    closest_header = max(preceding_headers, key=lambda n: self.reading_key(n.page_no, n.bbox))
                    ret_images[i] = image.model_copy(update={"level": closest_header.level + 1, "level_label": "Unreferenced Image", "parent_id": closest_header.snippet_id})

        return ret_images

    def _build_media_nodes(self, items: list[PictureItem] | list[TableItem], text_nodes: list[TextSnippetNode], node_cls: type[T], extra_fields: Callable[[Any], dict[str, Any]]) -> list[T]:
        """Shared builder for image and table nodes; parents and levels are
        resolved afterwards via rule-based caption matching."""
        nodes = []
        for i, item in enumerate(items):
            caption_nodes = self.get_caption_nodes(item, text_nodes)
            node = node_cls(
                snippet_id=item.self_ref,
                document_id=self._document_metadata.document_id,
                label=item.label,
                sequence_no=i,
                level=1000,  # placeholder, resolved by rb_image_parent_matching
                level_label="",
                parent_id=None,  # resolved by rb_image_parent_matching
                docling_parent_ref=item.parent.get_ref() if item.parent else None,
                docling_self_ref=item.get_ref(),
                is_grouped=item.parent is not None and "group" in item.parent.get_ref().cref,
                caption_nodes=caption_nodes,
                caption_text=" ".join([node.text for node in caption_nodes]) if len(caption_nodes) > 0 else "",
                bbox=item.prov[0].bbox,
                page_no=item.prov[0].page_no,
                provenance=provenance_from_prov(item.prov),
                **extra_fields(item),
            )
            nodes.append(node)
        # map parents and levels using rule-based matching
        return self.rb_image_parent_matching(nodes, text_nodes)

    def compute_image_nodes(self, images: list[PictureItem], text_nodes: list[TextSnippetNode]) -> list[ImageSnippetNode]:
        return self._build_media_nodes(images, text_nodes, ImageSnippetNode, lambda image: {})

    def compute_table_nodes(self, tables: list[TableItem], text_nodes: list[TextSnippetNode]) -> list[TableSnippetNode]:
        return self._build_media_nodes(tables, text_nodes, TableSnippetNode, lambda table: {
            "markdown_serialization": self.table_markdown(table),
            "html_serialization": table.export_to_html(self.docling_doc, add_caption=True),
            "continuation_refs": self._table_continuations.get(table.self_ref, []),
        })

    def table_markdown(self, table: TableItem) -> str:
        """The table body as markdown, without the caption.

        ``export_to_markdown(doc)`` prepends the caption whenever docling linked one
        itself, which would then appear twice: once from here and once from
        ``caption_text``, the node's own caption field (and the only place a caption
        recovered by :meth:`recover_caption_node` shows up at all). Stripping it here
        keeps ``caption_text`` the single source of the caption, whichever way it was
        found. Passing no ``doc`` would also drop it, but that call is deprecated."""
        markdown = table.export_to_markdown(self.docling_doc)
        caption = table.caption_text(self.docling_doc).strip()
        if caption and markdown.lstrip().startswith(caption):
            markdown = markdown.lstrip()[len(caption):]
        return markdown.lstrip("\n")

    def get_caption_nodes(self, image: PictureItem | TableItem, text_nodes: list[TextSnippetNode]) -> list[TextSnippetNode] | list:
        # find text node that contains reference to image or table
        caption_refs = image.captions if image.captions else []
        caption_nodes = []
        for caption_ref in caption_refs:
            for text_node in text_nodes:
                if caption_ref.get_ref().cref == text_node.docling_self_ref.get_ref().cref:
                    caption_nodes.append(text_node)
        if not caption_nodes and self.flags.recover_captions:
            # docling often detects the caption but leaves it as a plain text/list
            # item without linking it to the media item: recover it spatially
            recovered = self.recover_caption_node(image, text_nodes)
            if recovered is not None:
                caption_nodes.append(recovered)
        return caption_nodes

    def recover_caption_node(self, image: PictureItem | TableItem, text_nodes: list[TextSnippetNode]) -> TextSnippetNode | None:
        """Find an unlinked caption for a media item: caption-like text on the same
        page, vertically adjacent and horizontally overlapping (same column)."""
        bbox = image.prov[0].bbox
        page_no = image.prov[0].page_no
        candidates = []
        for node in text_nodes:
            if node.page_no != page_no or not CAPTION_PATTERN.match(node.text):
                continue
            # gap per axis between the two bboxes, 0 if they overlap on that axis
            h_gap = max(bbox.l - node.bbox.r, node.bbox.l - bbox.r, 0)
            v_gap = max(bbox.b - node.bbox.t, node.bbox.b - bbox.t, 0)
            above_or_below = h_gap == 0 and v_gap <= MAX_CAPTION_DISTANCE
            in_margin = v_gap == 0 and h_gap <= MAX_CAPTION_SIDE_GAP
            if above_or_below or in_margin:
                candidates.append((h_gap + v_gap, node))
        if not candidates:
            return None
        distance, best = min(candidates, key=lambda c: c[0])
        self.logger.info(f"Recovered caption {best.snippet_id} ({best.text[:60]!r}) for {image.self_ref} at distance {distance:.1f}pt.")
        return best

    def write_nx_graph(self, filename: str, text_nodes: list[TextSnippetNode], image_nodes: list[ImageSnippetNode], table_nodes: list[TableSnippetNode], edges: list[WeightedEdge], root_id: str = ROOT_NODE_ID, reference_edges: list[WeightedEdge] | None = None):
        nx_graph = self.build_nx_graph(text_nodes, image_nodes, table_nodes, edges, root_id, reference_edges)
        nx.write_gexf(nx_graph, filename, version="1.3")
        return nx_graph

    def build_nx_graph(self, text_nodes: list[TextSnippetNode], image_nodes: list[ImageSnippetNode], table_nodes: list[TableSnippetNode], edges: list[WeightedEdge], root_id: str = ROOT_NODE_ID, reference_edges: list[WeightedEdge] | None = None) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        if root_id == ROOT_NODE_ID:
            # synthetic root is not part of the node lists, add it explicitly
            nx_graph.add_node(ROOT_NODE_ID, label="document_root", level=-1, level_label="Root", text=self._document_metadata.title or "")
        for node in text_nodes:
            nx_graph.add_node(node.snippet_id, label=node.label, level=node.level, level_label=node.level_label, text=node.text, line_heights=str(node.line_heights), font_key=node.font_key or "", level_height=node.level_height if node.level_height is not None else -1.0, region=node.region)
        for node in image_nodes:
            # caption plus whatever is printed inside the figure: without the latter a
            # chart shows up in the viewer as a node with nothing in it
            nx_graph.add_node(node.snippet_id, label=node.label, level=node.level, level_label=node.level_label, text=node.content_text())
        for node in table_nodes:
            # caption only: a markdown table in a GEXF attribute is unreadable in a
            # graph viewer, and the json is the form that carries it
            nx_graph.add_node(node.snippet_id, label=node.label, level=node.level, level_label=node.level_label, text=node.caption_text)
        for parent_id, child_id, weight in edges:
            nx_graph.add_edge(parent_id, child_id, weight=weight, edge_type="hierarchy")
        for source_id, target_id, weight in reference_edges or []:
            # DiGraph holds one edge per node pair: never overwrite a hierarchy edge
            if not nx_graph.has_edge(source_id, target_id):
                nx_graph.add_edge(source_id, target_id, weight=weight, edge_type="reference")
        return nx_graph

    def absorb_figure_text(self, text_nodes: list[TextSnippetNode],
                           media_nodes: list[ImageSnippetNode]) -> list[TextSnippetNode]:
        """Fold a figure's own labels into the picture node they sit in.

        A chart is read line by line, so its axis ticks and data points arrive as one
        node each: "42", "33", "51". None of them is retrievable on its own -- there is
        nothing in "42" for a query to match, and nothing in it for a reader to use --
        while the picture that gives them their meaning carries no text at all and is
        therefore not retrievable either. Folding the labels into the picture fixes
        both ends: the figure becomes one unit holding its own contents.

        What is *not* absorbed is as important as what is. A node above
        ``figure_label_max_tokens`` is prose set inside a figure -- a step of a
        flowchart, a note in a boxed diagram -- which reads on its own and stays a node
        of its own, though it still appears in the digest so that the figure reads
        whole. A caption is never absorbed: it is already the media node's own field.
        Nor is a node that parents another, which absorbing would orphan.

        Runs after the media nodes have their captions and their parents, because both
        decide what may be taken.
        """
        if not self.flags.absorb_figure_text:
            return text_nodes

        captions = {caption.snippet_id for node in media_nodes for caption in node.caption_nodes}
        parents = {node.parent_id for node in text_nodes if node.parent_id}

        grouped: dict[str, list[TextSnippetNode]] = defaultdict(list)
        for node in text_nodes:
            if node.region == REGION_FIGURE and node.parent_id:
                grouped[node.parent_id].append(node)

        absorbed: set[str] = set()
        for media in media_nodes:
            members = sorted(grouped.get(media.snippet_id, []), key=lambda n: n.sequence_no)
            if not members:
                continue
            media.figure_text = FIGURE_TEXT_SEPARATOR.join(
                text for text in (member.text.strip() for member in members) if text)
            takeable = [member for member in members
                        if member.snippet_id not in captions
                        and member.snippet_id not in parents
                        and approximate_token_count(member.text) <= self.flags.figure_label_max_tokens]
            # only what is actually removed hands over its geometry: a label that stays
            # a node of its own still reports its own box, and would otherwise report it twice
            media.provenance += [entry for member in takeable for entry in member.provenance]
            absorbed.update(member.snippet_id for member in takeable)

        if absorbed:
            self.logger.info(f"Absorbed {len(absorbed)} figure labels into "
                             f"{sum(1 for m in media_nodes if m.figure_text)} media nodes.")
        return [node for node in text_nodes if node.snippet_id not in absorbed]

    def construct_snippet_nodes(self) -> tuple[list[TextSnippetNode], list[ImageSnippetNode], list[TableSnippetNode]]:
        text_nodes = self.compute_text_nodes(self.text_items)
        image_nodes = self.compute_image_nodes(self.image_items, text_nodes)
        table_nodes = self.compute_table_nodes(self.table_items, text_nodes) 
        text_nodes = self.absorb_figure_text(text_nodes, image_nodes + table_nodes)
        return text_nodes, image_nodes, table_nodes

    def resolve_root_id(self, text_nodes: list[TextSnippetNode]) -> str:
        """Return the snippet_id of the document title node to use as graph root.

        Prefers the node whose text matches the extracted document title, then any
        node labeled "Title"; falls back to the synthetic ROOT_NODE_ID if neither exists.
        """
        title = (self._document_metadata.title or "").strip().lower()
        if title:
            for node in text_nodes:
                if node.text.strip().lower() == title:
                    return node.snippet_id
        for node in text_nodes:
            if node.level_label == "Title":
                return node.snippet_id
        return ROOT_NODE_ID

    def edge_relation(self, parent: SnippetNode | None, child: SnippetNode) -> str:
        """The structural relation a parent->child edge realizes.

        Kept apart from the weight itself so every metric can read the relation:
        the rule-based weights look it up in the config, and the content metrics
        use it to tell a bullet from a subheading.
        """
        if isinstance(child, (ImageSnippetNode, TableSnippetNode)):
            if child.level_label == "Unreferenced Image":
                return RELATION_UNREFERENCED_MEDIA
            return RELATION_MEDIA
        if child.is_grouped:
            return RELATION_LIST_ITEM
        if child.level_label in ("Title", "Heading") and parent is not None and parent.level_label in ("Title", "Heading"):
            return RELATION_SECTION
        return RELATION_TEXT

    def compute_edge_weight(self, parent: SnippetNode | None, child: SnippetNode) -> float:
        """The structural weight of a parent->child edge."""
        return self.edge_weights.structural_weight(self.edge_relation(parent, child))

    def construct_snippet_edges(self, nodes: list[SnippetNode], root_id: str) -> list[WeightedEdge]:
        """Build weighted parent->child edges; nodes without a resolvable parent attach to the document root."""
        node_by_id = {node.snippet_id: node for node in nodes}
        edges = []
        for node in nodes:
            if node.snippet_id == root_id:
                continue  # the root node has no parent
            parent = node_by_id.get(node.parent_id) if node.parent_id else None
            if parent is not None and parent.snippet_id != node.snippet_id:
                relation = self.edge_relation(parent, node)
                edges.append((parent.snippet_id, node.snippet_id,
                              self.edge_weights.structural_weight(relation)))
                self._edge_relations[(parent.snippet_id, node.snippet_id)] = relation
            else:
                # missing or dangling parent_id: attach to the document root
                edges.append((root_id, node.snippet_id, self.edge_weights.root))
                self._edge_relations[(root_id, node.snippet_id)] = RELATION_ROOT
        return edges

    def connect_components(self, nodes: list[SnippetNode], edges: list[WeightedEdge], root_id: str) -> list[WeightedEdge]:
        """Attach every weakly connected component that does not reach the document root to it.

        construct_snippet_edges already links orphan nodes to the root, but parent
        cycles can still leave a component detached; this guarantees a single
        connected graph per document.
        """
        graph = nx.DiGraph()
        graph.add_node(root_id)
        graph.add_nodes_from(node.snippet_id for node in nodes)
        graph.add_edges_from((p, c) for p, c, _ in edges)
        node_by_id = {node.snippet_id: node for node in nodes}
        extra_edges = []
        for component in nx.weakly_connected_components(graph):
            if root_id in component:
                continue
            component_nodes = [node_by_id[n] for n in component if n in node_by_id]
            if not component_nodes:
                continue
            # attach the structurally highest node of the component to the root
            top = min(component_nodes, key=lambda n: (n.level, n.sequence_no))
            extra_edges.append((root_id, top.snippet_id, self.edge_weights.root))
            self._edge_relations[(root_id, top.snippet_id)] = RELATION_ROOT
        return extra_edges

    def construct_reference_edges(self, media_nodes: list[ImageSnippetNode | TableSnippetNode]) -> list[WeightedEdge]:
        """One edge per mentioning text snippet -> media item.

        These complement the hierarchy tree (where only the first mention
        determines the parent) and are kept separate from it: they may form
        cycles with tree edges and must not affect connectivity handling.
        """
        edges = []
        for node in media_nodes:
            for referencing_id in node.referencing_node_ids:
                edges.append((referencing_id, node.snippet_id, self.edge_weights.reference))
                self._edge_relations[(referencing_id, node.snippet_id)] = RELATION_REFERENCE
        return edges

    def node_texts(self, nodes: list[SnippetNode], root_id: str) -> dict[str, str]:
        """Text content per node id, used for content-based edge weights."""
        texts = {}
        if root_id == ROOT_NODE_ID:
            # synthetic root is not part of the node lists
            texts[ROOT_NODE_ID] = self._document_metadata.title or ""
        for node in nodes:
            texts[node.snippet_id] = (node.content_text() if isinstance(node, ImageSnippetNode)
                                      else node.text)
        return texts

    def build_edge_context(self, nodes: list[SnippetNode], root_id: str,
                           edges: list[Edge]) -> EdgeContext:
        """The document as an edge weight metric sees it.

        Carries more than the texts, because the need estimators ask what kind
        of unit a node is (a bullet, a heading, a table) and the surprisal
        control needs the tree shape to find a non-ancestor parent.
        """
        texts = self.node_texts(nodes, root_id)
        units: dict[str, Unit] = {}
        if root_id == ROOT_NODE_ID:
            units[ROOT_NODE_ID] = Unit(
                node_id=ROOT_NODE_ID, text=texts.get(ROOT_NODE_ID, ""),
                label="document_root", level_label="Root", level=-1, kind=KIND_ROOT,
            )
        for node in nodes:
            if isinstance(node, TableSnippetNode):
                kind = KIND_TABLE
            elif isinstance(node, ImageSnippetNode):
                kind = KIND_IMAGE
            else:
                kind = KIND_TEXT
            units[node.snippet_id] = Unit(
                node_id=node.snippet_id,
                text=texts.get(node.snippet_id, ""),
                label=node.label,
                level_label=node.level_label,
                level=node.level,
                sequence_no=node.sequence_no,
                is_grouped=node.is_grouped,
                kind=kind,
            )
        return EdgeContext(units=units, edges=list(edges), relations=dict(self._edge_relations))

    def get_graph(self, save_to="") -> SnippetGraph:
        self._edge_relations = {}
        text_nodes, image_nodes, table_nodes = self.construct_snippet_nodes()
        all_nodes: list[SnippetNode] = text_nodes + image_nodes + table_nodes
        root_id = self.resolve_root_id(text_nodes)
        edges = self.construct_snippet_edges(all_nodes, root_id)
        edges += self.connect_components(all_nodes, edges, root_id)
        reference_edges = self.construct_reference_edges(image_nodes + table_nodes)
        if self.edge_weights.metric != "structural" or self.edge_weights.combination != "replace":
            # weigh both edge kinds in one pass so they share a score normalization
            all_edges = edges + reference_edges
            ctx = self.build_edge_context(all_nodes, root_id, [(p, c) for p, c, _ in all_edges])
            combined = apply_edge_weights(all_edges, ctx, self.edge_weights)
            edges, reference_edges = combined[:len(edges)], combined[len(edges):]
        if save_to != "":
            self.write_nx_graph(save_to, text_nodes, image_nodes, table_nodes, edges, root_id, reference_edges)
        return SnippetGraph(text_nodes, image_nodes, table_nodes, edges, reference_edges, root_id,
                            self.page_geometry(), dict(self._edge_relations))