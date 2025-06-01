import math
import os
import pprint
from pathlib import Path

import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


def _calculate_midi_number(frequency: float) -> int:
    """Return the nearest MIDI note for a single frequency.
           We'll treat any non-positive freq as “no note” → MIDI 0."""
    if frequency <= 0.0:
        return 0
    num = 69 + 12 * math.log2(frequency / 440.0)
    return int(round(num))

def _convert(frequencies: np.ndarray,
             magnitudes:   np.ndarray) -> np.ndarray:
    """
    frequencies, magnitudes: both shape = (n_bins, n_frames)
    returns piano_roll: shape = (128, n_frames)
       where piano_roll[n, t] = max magnitude among all bins
       whose midi number == n in frame t.
    """
    vec_calc = np.vectorize(_calculate_midi_number, otypes=[int])
    midi = vec_calc(frequencies).astype(int)

    n_bins, n_frames = midi.shape
    midi_map = np.zeros((128, n_frames), dtype=float)

    # for each possible note 0...127, pick out its mags and max them
    for note in range(128):
        mask = (midi == note)
        # masked magnitudes, zeros outside
        masked = magnitudes * mask
        midi_map[note, :] = masked.max(axis=0)

    return midi_map


class FrequencyToMidi:
    def __init__(self, logger: HoornLogger, cache_dir: Path):
        self._logger = logger
        self._separator = self.__class__.__name__

        os.makedirs(cache_dir, exist_ok=True)
        self._convert = Memory(cache_dir, verbose=0).cache(_convert)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def convert(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        """Convert a 2D array of freqs → MIDI, using our scalar helper."""
        midi = self._convert(frequencies, magnitudes)

        self._logger.debug(f"Midi Shape: {midi.shape}", separator=self._separator)
        self._logger.debug(f"Midi:\n{pprint.pformat(midi)}", separator=self._separator)

        return midi



