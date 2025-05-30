from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from track_analysis.components.track_analysis.features.key_extraction.visualization.renderers.base_renderer import \
    Renderer


class BarRenderer(Renderer):
    def __init__(self):
        pass

    def render(self, ax: plt.Axes, data: np.ndarray) -> Any:
        x = np.arange(data.size)
        ax.bar(x, data.flatten())
