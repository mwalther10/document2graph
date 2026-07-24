from docling_core.types.doc.document import TextItem, PictureItem, TableItem, DoclingDocument # type: ignore
from docling_parse.pdf_parser import PdfDocument
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.page import TextCellUnit
import regex as re
import networkx as nx
from collections import Counter, defaultdict
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
from typing import Any, Callable, NamedTuple, TypeVar

from ..utils.edge_weight_lib import apply_relevancy_weights
from ..utils.log import Log
from .level_classifier import LevelClassifier
from .metadata_extractor import DocumentMetadataExtractor

T = TypeVar('T', bound=ImageSnippetNode)

ROOT_NODE_ID = "#/document-root"

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

SnippetNode = TextSnippetNode | ImageSnippetNode | TableSnippetNode
WeightedEdge = tuple[str, str, float]

class SnippetGraph(NamedTuple):
    """Result of get_graph: all snippet nodes, weighted edges, and the id of the root node.

    edges form the hierarchy tree (one parent per node); reference_edges link every
    text snippet that mentions a media item to that item and are not part of the tree.

    root_id is the snippet_id of the document title node, or the synthetic
    ROOT_NODE_ID (which is not part of the node lists) if no title node was found.
    """
    text_nodes: list[TextSnippetNode]
    image_nodes: list[ImageSnippetNode]
    table_nodes: list[TableSnippetNode]
    edges: list[WeightedEdge]
    reference_edges: list[WeightedEdge]
    root_id: str

class SnippetGraphConstructor():
    def __init__(self, pdf_doc: PdfDocument, docling_doc: DoclingDocument, filename: str, document_type: str, metadata_config: MetadataExtractionConfig | None = None, edge_weights: EdgeWeightConfig | None = None):
        self.pdf_doc = pdf_doc
        self.docling_doc = docling_doc
        self.edge_weights = edge_weights or EdgeWeightConfig()
        self.metadata_config = metadata_config or MetadataExtractionConfig()
        self.logger = Log("SnippetGraphConstructor").logger
        self._page_shapes_cache: dict[int, tuple[list[float], list[tuple[float, float, float, float]]]] = {}
        self.text_items = self.assign_regions(self.stitch_continuations(self.order_by_page_blocks(
            [TextSnippet(text_item=item, line_heights=self.add_line_heights(item), font_key=self.get_font_key(item)) for item in docling_doc.texts if item.label not in ("page_footer", "page_header")]
        )))
        self.table_items = docling_doc.tables
        self.image_items = self.filter_decorative_pictures(docling_doc.pictures)
        self._document_metadata = DocumentMetadataExtractor(self.text_items).extract(filename, document_type, self.metadata_config)
        self.levels = LevelClassifier(self.text_items, self._document_metadata.title)

    @property
    def document_metadata(self):
        return self._document_metadata

    def add_line_heights(self, snippet: TextItem) -> list[float]:
        snippet_bbox = snippet.prov[0].bbox
        lines = self.pdf_doc.get_page(snippet.prov[0].page_no).get_cells_in_bbox(cell_unit=TextCellUnit.LINE, bbox=snippet_bbox)
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
                if right - left >= BOX_MIN_WIDTH and top - bottom >= BOX_MIN_HEIGHT:
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
        """Whether a fragment breaks off mid-sentence: no terminal punctuation, ending on
        a lower case letter, a comma or a hyphen."""
        stripped = text.rstrip()
        if not stripped or TERMINAL_PUNCTUATION.search(text):
            return False
        return stripped[-1] in CONTINUATION_BREAKS or (stripped[-1].isalpha() and stripped[-1].islower())

    @staticmethod
    def _resumes(text: str) -> bool:
        """Whether a fragment picks up a sentence: it starts in lower case."""
        stripped = text.lstrip()
        return bool(stripped) and stripped[0].isalpha() and stripped[0].islower()

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
        chars = self.pdf_doc.get_page(snippet.prov[0].page_no).get_cells_in_bbox(cell_unit=TextCellUnit.CHAR, bbox=snippet.prov[0].bbox)
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
                # no caption or no match: assign to closest preceding heading
                preceding_headers = [node for node in text_nodes if node.level in self.levels.header_levels() and node.sequence_no < image.sequence_no]
                if preceding_headers:
                    lowest_level_header = min(preceding_headers, key=lambda n: n.level)
                    ret_images[i] = image.model_copy(update={"level": lowest_level_header.level + 1, "level_label": "Unreferenced Image", "parent_id": lowest_level_header.snippet_id})

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
                **extra_fields(item),
            )
            nodes.append(node)
        # map parents and levels using rule-based matching
        return self.rb_image_parent_matching(nodes, text_nodes)

    def compute_image_nodes(self, images: list[PictureItem], text_nodes: list[TextSnippetNode]) -> list[ImageSnippetNode]:
        return self._build_media_nodes(images, text_nodes, ImageSnippetNode, lambda image: {})

    def compute_table_nodes(self, tables: list[TableItem], text_nodes: list[TextSnippetNode]) -> list[TableSnippetNode]:
        return self._build_media_nodes(tables, text_nodes, TableSnippetNode, lambda table: {
            "markdown_serialization": table.export_to_markdown(self.docling_doc),
            "html_serialization": table.export_to_html(self.docling_doc, add_caption=True),
        })

    def get_caption_nodes(self, image: PictureItem | TableItem, text_nodes: list[TextSnippetNode]) -> list[TextSnippetNode] | list:
        # find text node that contains reference to image or table
        caption_refs = image.captions if image.captions else []
        caption_nodes = []
        for caption_ref in caption_refs:
            for text_node in text_nodes:
                if caption_ref.get_ref().cref == text_node.docling_self_ref.get_ref().cref:
                    caption_nodes.append(text_node)
        if not caption_nodes:
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
        for node in image_nodes + table_nodes:
            nx_graph.add_node(node.snippet_id, label=node.label, level=node.level, level_label=node.level_label, text=node.caption_text)
        for parent_id, child_id, weight in edges:
            nx_graph.add_edge(parent_id, child_id, weight=weight, edge_type="hierarchy")
        for source_id, target_id, weight in reference_edges or []:
            # DiGraph holds one edge per node pair: never overwrite a hierarchy edge
            if not nx_graph.has_edge(source_id, target_id):
                nx_graph.add_edge(source_id, target_id, weight=weight, edge_type="reference")
        return nx_graph

    def construct_snippet_nodes(self) -> tuple[list[TextSnippetNode], list[ImageSnippetNode], list[TableSnippetNode]]:
        text_nodes = self.compute_text_nodes(self.text_items)
        image_nodes = self.compute_image_nodes(self.image_items, text_nodes)
        table_nodes = self.compute_table_nodes(self.table_items, text_nodes) 
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

    def compute_edge_weight(self, parent: SnippetNode | None, child: SnippetNode) -> float:
        weights = self.edge_weights
        if isinstance(child, (ImageSnippetNode, TableSnippetNode)):
            if child.level_label == "Unreferenced Image":
                return weights.unreferenced_media
            return weights.media
        if child.is_grouped:
            return weights.list_item
        if child.level_label in ("Title", "Heading") and parent is not None and parent.level_label in ("Title", "Heading"):
            return weights.section
        return weights.text

    def construct_snippet_edges(self, nodes: list[SnippetNode], root_id: str) -> list[WeightedEdge]:
        """Build weighted parent->child edges; nodes without a resolvable parent attach to the document root."""
        node_by_id = {node.snippet_id: node for node in nodes}
        edges = []
        for node in nodes:
            if node.snippet_id == root_id:
                continue  # the root node has no parent
            parent = node_by_id.get(node.parent_id) if node.parent_id else None
            if parent is not None and parent.snippet_id != node.snippet_id:
                edges.append((parent.snippet_id, node.snippet_id, self.compute_edge_weight(parent, node)))
            else:
                # missing or dangling parent_id: attach to the document root
                edges.append((root_id, node.snippet_id, self.edge_weights.root))
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
        return edges

    def node_texts(self, nodes: list[SnippetNode], root_id: str) -> dict[str, str]:
        """Text content per node id, used for content-based relevancy weights."""
        texts = {}
        if root_id == ROOT_NODE_ID:
            # synthetic root is not part of the node lists
            texts[ROOT_NODE_ID] = self._document_metadata.title or ""
        for node in nodes:
            if isinstance(node, TableSnippetNode):
                texts[node.snippet_id] = f"{node.caption_text}\n{node.markdown_serialization}".strip()
            elif isinstance(node, ImageSnippetNode):
                texts[node.snippet_id] = node.caption_text
            else:
                texts[node.snippet_id] = node.text
        return texts

    def get_graph(self, save_to="") -> SnippetGraph:
        text_nodes, image_nodes, table_nodes = self.construct_snippet_nodes()
        all_nodes: list[SnippetNode] = text_nodes + image_nodes + table_nodes
        root_id = self.resolve_root_id(text_nodes)
        edges = self.construct_snippet_edges(all_nodes, root_id)
        edges += self.connect_components(all_nodes, edges, root_id)
        reference_edges = self.construct_reference_edges(image_nodes + table_nodes)
        if self.edge_weights.relevancy.enabled:
            # apply in one pass so both edge kinds share the same score normalization
            combined = apply_relevancy_weights(edges + reference_edges, self.node_texts(all_nodes, root_id), self.edge_weights.relevancy)
            edges, reference_edges = combined[:len(edges)], combined[len(edges):]
        if save_to != "":
            self.write_nx_graph(save_to, text_nodes, image_nodes, table_nodes, edges, root_id, reference_edges)
        return SnippetGraph(text_nodes, image_nodes, table_nodes, edges, reference_edges, root_id)