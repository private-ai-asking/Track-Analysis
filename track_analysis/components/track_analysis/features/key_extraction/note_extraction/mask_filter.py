import numpy as np
from scipy.ndimage import median_filter

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class MaskFilter:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator: str = "MaskFilter"

    def filter(
            self,
            mask: np.ndarray,
            hop_length: int,
            sample_rate: int,
            tempo: float,
            segment_level: int
    ) -> np.ndarray:
        quarter_note_duration_seconds = 60.0 / tempo

        # Map beat level → fraction of a bar for the *minimum* duration
        #   4 → 1/4-bar = quarter note
        #   3 → 1/8-bar = eighth note
        #   2 → 1/16-bar = sixteenth note
        #   1 → 1/32-bar
        #   0 → 1/64-bar
        fraction_of_bar = {
            4: 1/4,
            3: 1/8,
            2: 1/16,
            1: 1/32,
            0: 1/64
        }[segment_level]

        time_threshold_s = quarter_note_duration_seconds * 4 * fraction_of_bar
        time_threshold_ms = time_threshold_s * 1000

        self._logger.debug(f"Chosen time threshold: {time_threshold_ms:.3f}ms", separator=self._separator)

        frame_ms = hop_length / sample_rate * 1000
        smoothing_window_frame_size = int(np.ceil(time_threshold_ms/frame_ms))

        self._logger.debug(f"Smoothing window: {smoothing_window_frame_size} frames (~{time_threshold_ms:.0f}ms)", separator=self._separator)

        out = np.zeros_like(mask)
        for pc in range(mask.shape[0]):
            out[pc] = median_filter(mask[pc].astype(int), size=smoothing_window_frame_size) > 0

        for p in range(12):
            self._logger.debug(f"Class {p:2d}: {out[p].sum()} active frames (after median_filter)", separator=self._separator)

        return out
