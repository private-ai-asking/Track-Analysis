from typing import Optional

import numpy as np

from track_analysis.components.track_analysis.features.key_extraction.utils.convert_to_db import convert_to_db_func
from track_analysis.components.track_analysis.features.key_extraction.feature.visualization.renderers.base_renderer import \
    Renderer


class PianoRollRenderer(Renderer):
    def __init__(self, cmap="viridis", min_midi=0, num_ticks: Optional[int] = None, min_v: int = -60, max_v: int = 0, convert_to_db: bool = True):
        self._cmap = cmap
        self._min = min_midi
        self._num_ticks = num_ticks
        self._min_v = min_v
        self._max_v = max_v

        self._convert_to_db = convert_to_db
        self._compute_db = convert_to_db_func

    def render(self, ax, midi_map: np.ndarray):
        """
        midi_map: shape = (n_bins, n_frames), values = MIDI numbers
        """
        if self._convert_to_db:
            midi_map = self._compute_db(midi_map, ref=np.max)

        n_bins, n_frames = midi_map.shape
        extent = [0, n_frames,
                  self._min, self._min+n_bins]

        if self._num_ticks:
            ax.set(yticks=np.arange(self._num_ticks))
        img = ax.imshow(
            midi_map,
            origin="lower",
            aspect="auto",
            cmap=self._cmap,
            extent=extent,  # type: ignore
            interpolation="nearest",
            vmin=self._min_v,
            vmax=self._max_v,
        )
        return img
