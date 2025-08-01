import pprint

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.cacheing.shared import MEMORY


@MEMORY.cache()
def _convert_to_chroma(midi: np.ndarray) -> np.ndarray:
    """
    Cached conversion from MIDI piano-roll to chroma features.
    """
    n_notes, n_frames = midi.shape
    chroma = np.zeros((12, n_frames), dtype=float)
    note_indices = np.arange(n_notes)

    for pc in range(12):
        mask = (note_indices % 12) == pc
        chroma[pc] = midi[mask].sum(axis=0)

    return chroma

class MidiToPitchClassesConverter:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def convert(self, midi: np.ndarray) -> np.ndarray:
        chroma = _convert_to_chroma(midi=midi)
        self._logger.debug(f"Pitch Classes Shape: {chroma.shape}", separator=self._separator)
        self._logger.debug(f"Pitch Classes:\n{pprint.pformat(chroma)}", separator=self._separator)
        return chroma
