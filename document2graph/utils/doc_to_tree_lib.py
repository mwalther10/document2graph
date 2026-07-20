import numpy as np

# schemas
from ..models.TextSnippet import TextSnippet

def get_heights(text_items: list[TextSnippet]) -> np.ndarray:
    heights = [height for snippet in text_items for height in snippet.line_heights]
    return np.multiply(np.round(np.array(heights), 1), 10)

def get_height_hist(heights=np.array, plot=False) -> dict[int, int]:
        if not heights.size:
            print("No heights found, returning empty histogram.")
            return {}
        unique_heights, counts = np.unique(heights, return_counts=True)
        height_hist = dict(zip(unique_heights, counts))
        if plot:
            # matplotlib is only needed for this debug plot, not a package dependency
            import matplotlib.pyplot as plt
            plt.bar(height_hist.keys(), height_hist.values()) # type: ignore
            xticks = unique_heights[::max(1, len(unique_heights)//20)]
            plt.xticks(xticks, rotation=90, size=6)
            plt.xlabel("Font size")
            plt.ylabel("Count")
            plt.show()
            
        return height_hist

def get_text_body_height(heights: np.ndarray) -> tuple[float, float]:
        assert heights.size > 0, "No heights provided to compute text body height."
        # text body level range: between 25th and 75th percentile
        text_body_height = np.percentile(heights, 50)  # median
        iqr = np.percentile(heights, 75) - np.percentile(heights, 25)
        return float(text_body_height), float(iqr)

def line_height_to_key(heights: list[float]) -> np.ndarray:
    return np.multiply(heights, 10).astype(int)  # multiply by 10 to avoid float precision issues

def compute_snippet_level(snippet: TextSnippet, levels: dict[int, int]) -> int:
    heights = line_height_to_key(snippet.line_heights)
    if not heights.size:
        return 1000  # unknown level
    # get median height in case there are multiple lines with different heights
    median_height = np.median(heights)
    if median_height in levels:
        return levels[int(median_height)]
    else:
        # find closest height in levels
        closest_height = min(levels.keys(), key=lambda h: abs(h - median_height))
        return levels[closest_height]