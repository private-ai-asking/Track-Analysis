from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.feature.visualization.renderers.base_renderer import \
    Renderer


class HeatmapRenderer(Renderer):
    def __init__(self, cmap: str = "viridis", num_ticks: int = 12):
        self._cmap = cmap
        self._num_ticks = num_ticks

    def render(self, ax: plt.Axes, data: np.ndarray) -> Any:
        frames = data.shape[1]
        # noinspection PyTypeChecker
        img = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[0, frames, 0, data.shape[0]],
            cmap=self._cmap,
        )
        ax.set(yticks=np.arange(self._num_ticks))
        return img
