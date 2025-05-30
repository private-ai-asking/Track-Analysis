import os
import pprint
from pathlib import Path

import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

def _calculate_pitch_class(midi_number: int) -> int:
    return midi_number % 12

def _convert(midi: np.ndarray) -> np.ndarray:
    n_notes, n_frames = midi.shape
    chroma = np.zeros((12, n_frames), dtype=float)
    note_indices = np.arange(n_notes)

    for pc in range(12):
        mask = (note_indices % 12) == pc
        chroma[pc] = midi[mask].sum(axis=0)

    return chroma

class MidiToPitchClassesConverter:
    def __init__(self, logger: HoornLogger, cache_dir: Path):
        self._logger = logger
        self._separator = self.__class__.__name__

        os.makedirs(cache_dir, exist_ok=True)
        self._convert = Memory(cache_dir, verbose=0).cache(_convert)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def convert(self, midi: np.ndarray) -> np.ndarray:
        converted = self._convert(midi)

        self._logger.debug(f"Pitch Classes Shape: {converted.shape}", separator=self._separator)
        self._logger.debug(f"Pitch Classes:\n{pprint.pformat(midi)}", separator=self._separator)

        return converted

