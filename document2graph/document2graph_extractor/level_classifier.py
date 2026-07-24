import bisect
import numpy as np
import networkx as nx
from collections import defaultdict

from docling_core.types.doc.document import SectionHeaderItem # type: ignore

from ..models.TextSnippet import (
    REGION_BODY,
    REGION_FIGURE,
    REGION_FRONT_MATTER,
    REGION_SIDEBAR,
    TextSnippet,
)
from ..utils.doc_to_tree_lib import get_heights, get_height_hist, get_text_body_height, compute_snippet_level

# Two header heights count as different sizes only across a real size step: a gap
# larger than max(HEIGHT_STEP_FLOOR pt, HEIGHT_STEP_RATIO * larger height). Below
# that, differences are treated as measurement noise and stay in one height cluster.
HEIGHT_STEP_FLOOR = 1.5
HEIGHT_STEP_RATIO = 0.10

# A smaller style is placed above a larger one only on overwhelming evidence: its
# enclosure score (see _enclosure) must reach OVERRIDE_MIN and be OVERRIDE_RATIO
# times the score of the reverse direction, spread over at least OVERRIDE_MIN_SECTIONS
# of its sections, and it must hold OVERRIDE_MIN_SHARE of the larger style's headings.
OVERRIDE_MIN = 0.5
OVERRIDE_RATIO = 5.0
OVERRIDE_MIN_SECTIONS = 3
OVERRIDE_MIN_SHARE = 0.7

# a header style: its embedded-font key (None if the document carries no font info)
# and the representative height of its height cluster
Style = tuple[str | None, float]

# (representative height, level) for each height cluster of a font
FontClusters = list[tuple[float, int]]


class LevelClassifier:
    """Derives hierarchy levels for the document tree.

    The outline is built from the body alone. Docling marks the title of a boxed
    sidebar, a label inside a figure and the masthead of the first page as section
    headers just like a real section heading, and a single one of them ranked too high
    drags every section under it, so they are placed by their region (see
    SnippetGraphConstructor.assign_regions) instead of taking part in the ranking.

    Body headers are grouped into *styles*: one per (embedded-font key, height cluster)
    pair. Neither signal ranks them on its own -- font size can invert the hierarchy
    (some documents set subsections in a taller face than the sections that contain
    them), and a single font is often used for two different levels -- but the pair
    identifies a heading style reliably.

    Styles are then ordered by how they nest in reading order: for every pair, the
    style whose sections enclose the other one's headers is the parent. Enclosure is
    measured only where the document actually shows it, so a style whose first sections
    have no subsections at all (very common: the opening sections of a document are
    flat) is not pushed down by them; and a style can only hold headers that follow it,
    so a note recurring through the document cannot become their parent by seeing them
    between its own occurrences. Where two styles never nest, a real size step decides,
    and overturning that step takes enclosure evidence from several sections. Where
    neither signal says anything the styles are left unordered. Levels are the longest
    chain of parents above a style, so styles that relate to nothing stay siblings of
    whatever they cannot be ranked against instead of stacking a spurious hierarchy.

    Body text and sub-/footnote text are ranked by font height below the header levels."""

    def __init__(self, text_items: list[TextSnippet], title: str | None = None):
        self.text_items = text_items
        self.title = title
        self.heights = get_heights(text_items)
        self._has_title = False
        # representative height per header level, used only to place headers with no font info
        self._header_level_height: dict[int, float] = {}
        self._style_by_height: dict[tuple[str | None, float], Style] = {}
        self._level_by_style: dict[Style, int] = {}
        self._outline_top = 0
        self._aside_level = 0
        self.font_to_clusters, self.section_level_to_label = self._compute_section_header_levels()
        # body levels continue below the deepest header level -- the outline levels, the
        # title and the level asides are placed on
        num_section_headers = max([*self._header_level_height, self._aside_level,
                                   *([0] if self._has_title else [])], default=-1) + 1
        self.text_body_levels, self.text_level_to_label = self._compute_text_body_levels(num_section_headers)

    def classify(self, snippet: TextSnippet) -> tuple[int, str]:
        """Return (level, level_label) for a text snippet."""
        if isinstance(snippet.text_item, SectionHeaderItem):
            if snippet.text_item.text == self.title:
                return 0, "Title"
            level = self._header_level(snippet)
            return level, self.section_level_to_label.get(level, "Heading")
        level = compute_snippet_level(snippet, self.text_body_levels)
        return level, self.text_level_to_label.get(level, "unknown")

    def header_levels(self) -> set[int]:
        """Levels that belong to section headers (incl. the title)."""
        return set(self.section_level_to_label.keys())

    def _header_level(self, snippet: TextSnippet) -> int:
        """Level of a section header.

        Only body headers are ranked into the outline. The masthead sits at the top
        outline level, where its headings are peers of the sections rather than their
        parents, and the headings of an aside (a boxed sidebar, a figure label) sit
        below the deepest section so they always stay leaves."""
        if snippet.region == REGION_FRONT_MATTER:
            return self._outline_top
        if snippet.region in (REGION_SIDEBAR, REGION_FIGURE):
            return self._aside_level
        style = self._style_of(snippet.font_key, self._snippet_height(snippet), self._style_by_height)
        level = self._level_by_style.get(style) if style is not None else None
        # a header with no/unknown font key, or of a style no body header uses: place it
        # by height among the outline levels instead
        return level if level is not None else self._fallback_header_level(snippet)

    def _compute_section_header_levels(self) -> tuple[dict[str | None, FontClusters], dict[int, str]]:
        section_headers = [s for s in self.text_items if isinstance(s.text_item, SectionHeaderItem)]

        # The title always occupies the top level (0) via classify(); real section
        # headers rank below it, starting at level 1. Title snippets are excluded from
        # the ranking so the title's own style never opens a header tier of its own.
        has_title = bool(self.title) and any(s.text_item.text == self.title for s in section_headers)
        self._has_title = has_title
        ranked_headers = [s for s in section_headers if not (has_title and s.text_item.text == self.title)]
        start_level = 1 if has_title else 0

        level_to_label: dict[int, str] = {}
        if has_title:
            level_to_label[0] = "Title"

        # The outline is carried by the body alone; a masthead heading, the title of a
        # boxed sidebar and a label inside a figure are section headers to docling but
        # are no part of the document's section structure.
        styles, self._style_by_height = self._build_styles(ranked_headers)
        outline = [style for snippet, style in zip(ranked_headers, styles)
                   if style is not None and snippet.region == REGION_BODY]
        if not outline:
            return {}, level_to_label

        style_to_level = self._rank_styles(outline)
        font_to_clusters: dict[str | None, FontClusters] = defaultdict(list)
        heights_per_level: dict[int, list[float]] = defaultdict(list)
        for style, rank in style_to_level.items():
            font, rep = style
            level = start_level + rank
            font_to_clusters[font].append((rep, level))
            heights_per_level[level].append(rep)
            level_to_label[level] = "Heading"
        self._level_by_style = {style: start_level + rank for style, rank in style_to_level.items()}

        # Headers outside the outline get a level without taking part in the ranking:
        # the masthead sits at the top level (its headings are peers of the sections),
        # asides sit one level below the deepest section so they stay leaves.
        self._outline_top = start_level
        self._aside_level = start_level + max(style_to_level.values()) + 1
        level_to_label[self._aside_level] = "Heading"
        # several styles can share a level; keep one representative height per level. The
        # aside level has no height of its own -- headers land there by region, and a
        # header of unknown size should fall back into the outline, not below it.
        self._header_level_height = {lvl: float(np.median(reps)) for lvl, reps in heights_per_level.items()}
        return dict(font_to_clusters), level_to_label

    def _build_styles(self, headers: list[TextSnippet]) -> tuple[list[Style | None], dict[tuple[str | None, float], Style]]:
        """Split the headers into styles -- one per (font, height cluster) -- and return
        the style of every header in reading order plus a (font, height) -> style lookup.

        Height clusters are built from *every* header of a font, including the ones that
        are later left out of the outline: dropping them first would move the cluster
        boundaries and reshuffle the headers that remain. A header belongs to the cluster
        that contains its height, so a header near a boundary cannot drift into a
        neighbouring cluster when that cluster's median moves.

        Headers with no font key are left out when the document has font information at
        all; classify() places those by height instead. When no header carries a font key,
        every header is grouped under the None font, which reduces styles to pure height
        clusters."""
        use_font = any(s.font_key is not None for s in headers)
        heights_by_font: dict[str | None, list[float]] = defaultdict(list)
        for snippet in headers:
            if use_font and snippet.font_key is None:
                continue
            heights_by_font[snippet.font_key if use_font else None].append(self._snippet_height(snippet))

        style_by_height: dict[tuple[str | None, float], Style] = {}
        for font, heights in heights_by_font.items():
            for cluster in self._cluster_heights(heights):
                rep = float(np.median(cluster))
                for height in cluster:
                    style_by_height[(font, height)] = (font, rep)

        styles: list[Style | None] = []
        for snippet in headers:
            font = snippet.font_key if use_font else None
            styles.append(self._style_of(font, self._snippet_height(snippet), style_by_height))
        return styles, style_by_height

    @staticmethod
    def _style_of(font: str | None, height: float, style_by_height: dict[tuple[str | None, float], Style]) -> Style | None:
        """The style a header belongs to: the cluster of its font that contains its
        height, or the nearest one if the height was never clustered."""
        style = style_by_height.get((font, round(height, 1)))
        if style is not None:
            return style
        reps = {s for (f, _), s in style_by_height.items() if f == font}
        return min(reps, key=lambda s: abs(s[1] - height)) if reps else None

    def _rank_styles(self, seq: list[Style]) -> dict[Style, int]:
        """Rank the header styles into levels (0 = top) from their nesting in reading order.

        For every pair of styles the parent is the one that encloses the other; a real
        size step decides pairs that never nest, and pairs with neither signal are left
        unordered so they end up on the same level. The level of a style is the longest
        chain of parents above it."""
        styles = list(dict.fromkeys(seq))
        positions = {style: [i for i, s in enumerate(seq) if s == style] for style in styles}
        evidence = {(a, b): self._enclosure(positions[a], positions[b])
                    for a in styles for b in styles if a != b}
        enclosed = {(a, b): self._enclosed_share(positions[a], positions[b])
                    for a in styles for b in styles if a != b}
        sections = {style: len(positions[style]) - 1 for style in styles}

        graph = nx.DiGraph()
        graph.add_nodes_from(styles)
        for i, a in enumerate(styles):
            for b in styles[i + 1:]:
                tall, short = (a, b) if a[1] >= b[1] else (b, a)
                if self._is_size_step(tall[1], short[1]):
                    # a real size step: the taller style is the parent unless the smaller
                    # one demonstrably wraps it (small-caps section headings above larger
                    # subsection headings, say). Overturning the sizes takes evidence from
                    # several sections that really do hold most of the larger headings --
                    # one wide-spanning pair of headings is a coincidence, not a chapter.
                    inverted = (sections[short] >= OVERRIDE_MIN_SECTIONS
                                and enclosed[(short, tall)] >= OVERRIDE_MIN_SHARE
                                and evidence[(short, tall)] >= OVERRIDE_MIN
                                and evidence[(short, tall)] > OVERRIDE_RATIO * evidence[(tall, short)])
                    parent, child = (short, tall) if inverted else (tall, short)
                else:
                    # Same apparent size: whichever encloses the other is the parent, but
                    # a style can only hold headings that follow it -- a subsection never
                    # precedes every section of its document.
                    first = {style: positions[style][0] for style in (a, b)}
                    candidates = [(p, c) for p, c in ((a, b), (b, a))
                                  if first[p] < first[c] and evidence[(p, c)] > 0]
                    if not candidates:
                        continue  # no evidence either way: siblings
                    parent, child = max(candidates, key=lambda pair: evidence[pair])
                # how far to trust this edge if it has to be broken up later: an edge
                # backed by enclosure evidence outranks one that only follows the sizes
                margin = evidence[(parent, child)] - evidence[(child, parent)]
                confidence = (True, margin) if margin > 0 else (False, tall[1] - short[1])
                graph.add_edge(parent, child, confidence=confidence)

        self._break_cycles(graph)
        levels: dict[Style, int] = {}
        for style in nx.topological_sort(graph):
            levels[style] = max((levels[parent] + 1 for parent in graph.predecessors(style)), default=0)
        return levels

    @staticmethod
    def _break_cycles(graph: nx.DiGraph) -> None:
        """Make the style order acyclic by dropping the least supported edge of each
        cycle (pairwise comparisons are independent and can contradict each other)."""
        while True:
            try:
                cycle = nx.find_cycle(graph)
            except nx.NetworkXNoCycle:
                return
            parent, child, *_ = min(cycle, key=lambda edge: graph[edge[0]][edge[1]]["confidence"])
            graph.remove_edge(parent, child)

    @staticmethod
    def _enclosed_share(parent_positions: list[int], child_positions: list[int]) -> float:
        """Share of the child style's headings that sit inside a section of the parent
        style. Low when the two styles merely alternate: a note that recurs through the
        document sees headings between its own occurrences without holding them."""
        sections = len(parent_positions) - 1
        if sections < 1 or not child_positions:
            return 0.0
        first, last = parent_positions[0], parent_positions[-1]
        enclosed = bisect.bisect_left(child_positions, last) - bisect.bisect_right(child_positions, first)
        return enclosed / len(child_positions)

    @staticmethod
    def _enclosure(parent_positions: list[int], child_positions: list[int]) -> float:
        """How strongly one style's sections enclose another style's headers in reading
        order: average number of enclosed headers per section, weighted by the share of
        the child style's headers that are enclosed at all.

        Only the sections *between* two headers of the parent style count. The text
        before the first and after the last one is not attributable to it -- counting it
        would let a run of front-matter headings appear to enclose the whole document.
        Sections that contain no subsection at all cost nothing beyond diluting the
        average, so a document whose later sections are the only nested ones is still
        ranked correctly."""
        sections = len(parent_positions) - 1
        if sections < 1 or not child_positions:
            return 0.0
        # positions are unique per style, so "inside a section" is simply "between the
        # first and the last header of the parent style"
        first, last = parent_positions[0], parent_positions[-1]
        enclosed = bisect.bisect_left(child_positions, last) - bisect.bisect_right(child_positions, first)
        return (enclosed / sections) * (enclosed / len(child_positions))

    @staticmethod
    def _is_size_step(taller: float, shorter: float) -> bool:
        """Whether two heights differ by a real size step rather than measurement noise."""
        return taller - shorter > max(HEIGHT_STEP_FLOOR, HEIGHT_STEP_RATIO * taller)

    @staticmethod
    def _snippet_height(snippet: TextSnippet) -> float:
        """Representative font height of a snippet (median across its lines)."""
        return float(np.median(snippet.line_heights)) if snippet.line_heights else 0.0

    @classmethod
    def _cluster_heights(cls, heights: list[float]) -> list[list[float]]:
        """Split header heights into descending clusters, breaking only on a real size
        step so measurement noise stays in one cluster. Returns clusters largest-first."""
        unique = sorted({round(h, 1) for h in heights if h > 0}, reverse=True)
        if not unique:
            return []
        clusters = [[unique[0]]]
        for h in unique[1:]:
            if cls._is_size_step(clusters[-1][0], h):
                clusters.append([h])
            else:
                clusters[-1].append(h)
        return clusters

    def _fallback_header_level(self, snippet: TextSnippet) -> int:
        """Level for a header whose font key is missing: the header level whose
        representative height is closest, falling back to the top header level."""
        if not self._header_level_height:
            return 1 if self._has_title else 0
        height = self._snippet_height(snippet)
        return min(self._header_level_height, key=lambda lvl: abs(self._header_level_height[lvl] - height))

    def level_height(self, level: int) -> float | None:
        """Representative font height of a header sub-level, for debugging the level
        assignment. None for the title, body, and unknown levels."""
        return self._header_level_height.get(level)


    def _compute_text_body_levels(self, num_section_headers: int) -> tuple[dict[int, int], dict[int, str]]:
        all_heights = self.heights
        if not all_heights.size:
            return {}, {}

        texts = [s for s in self.text_items if not isinstance(s.text_item, SectionHeaderItem) and not (s.text_item.label == "page_footer" or s.text_item.label == "page_header")]

        text_body_heights = get_heights(texts) 
        text_body_height_hist = get_height_hist(text_body_heights)
        unique_text_body_heights = sorted(text_body_height_hist.keys(), reverse=True)
        text_body_height, iqr = get_text_body_height(all_heights)
        subtext_heights = [h for h in all_heights if h < text_body_height - iqr]
        subtext_median = np.percentile(subtext_heights, 50) if subtext_heights else 0

        level_dict = defaultdict(int)
        level_to_label = defaultdict(str)
        level = num_section_headers  # continue after section header levels
        subtexts = [h for h in unique_text_body_heights if h < text_body_height - iqr]

        for h in unique_text_body_heights:
            if(h >= text_body_height - iqr):
                level_dict[int(h)] = level  # body text
                level_to_label[ level ] = "Body"
            else:
                level += 1
                break

        for h in subtexts:
            if(h >= subtext_median - iqr and h <= subtext_median + iqr):
                level_dict[int(h)] = level  # subtext
                level_to_label[ level ] = "Subtext"
            else:
                level += 1
                level_dict[int(h)] = level  # smaller subtext, footnotes, etc.
                level_to_label[ level ] = "Subsubtext"

        return level_dict, level_to_label
