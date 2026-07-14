import numpy as np
from collections import defaultdict

from docling_core.types.doc.document import SectionHeaderItem

from ..models.TextSnippet import TextSnippet
from ..utils.doc_to_tree_lib import get_heights, get_height_hist, get_text_body_height, compute_snippet_level


class LevelClassifier:
    """Derives hierarchy levels from font heights: section headers get the top levels
    (largest fonts first), body text and sub-/footnote text continue below them."""

    def __init__(self, text_items: list[TextSnippet]):
        self.text_items = text_items
        self.heights = get_heights(text_items, skip_first_page=True)
        self.section_header_levels, self.section_level_to_label = self._compute_section_header_levels()
        num_section_headers = max(self.section_header_levels.values(), default=-1) + 1
        self.text_body_levels, self.text_level_to_label = self._compute_text_body_levels(num_section_headers)

    def classify(self, snippet: TextSnippet) -> tuple[int, str]:
        """Return (level, level_label) for a text snippet."""
        if isinstance(snippet.text_item, SectionHeaderItem):
            level = compute_snippet_level(snippet, self.section_header_levels)
            return level, self.section_level_to_label.get(level, "unknown")
        level = compute_snippet_level(snippet, self.text_body_levels)
        return level, self.text_level_to_label.get(level, "unknown")

    def header_levels(self) -> set[int]:
        """Levels that belong to section headers (incl. the title)."""
        return set(self.section_header_levels.values())

    def _compute_section_header_levels(self) -> tuple[dict[int, int], dict[int, str]]:
        all_heights = self.heights
        if not all_heights.size:
            return {}, {}
        section_headers = [s for s in self.text_items if isinstance(s.text_item, SectionHeaderItem)]
        section_header_height_hist = get_height_hist(get_heights(section_headers, skip_first_page=True))
        unique_section_header_heights = sorted(section_header_height_hist.keys(), reverse=True)
        text_body_height, iqr = get_text_body_height(all_heights)

        # group levels that are very close to each other on text body level (within iqr)
        level_dict = defaultdict(int)
        level_to_label = defaultdict(str)
        level = 0
        for h in unique_section_header_heights:
            if(h >= text_body_height - iqr and h <= text_body_height + iqr):
                level_dict[int(h)] = level  # body text headers, usually just bold printed text
                level_to_label[level] = "Heading"
            else:
                level_dict[int(h)] = level
                level_to_label[level] = "Heading"
                level += 1

        # check if largest height is unique -> title
        if len(unique_section_header_heights) > 0 and (section_header_height_hist[unique_section_header_heights[0]] == 1):
            level_to_label[0] = "Title"

        return level_dict, level_to_label

    def _compute_text_body_levels(self, num_section_headers: int) -> tuple[dict[int, int], dict[int, str]]:
        all_heights = self.heights
        if not all_heights.size:
            return {}, {}

        texts = [s for s in self.text_items if not isinstance(s.text_item, SectionHeaderItem)]

        text_body_heights = get_heights(texts, skip_first_page=True)
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
