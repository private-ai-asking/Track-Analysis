import math
from pathlib import Path
import numpy as np
import librosa

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["frequencies", "magnitudes"])
def _convert_to_midi(
        *,
        file_path: Path,
        sample_rate: int,
        frequencies: np.ndarray = None,
        magnitudes: np.ndarray = None,
) -> np.ndarray:
    """
    Cached frequency-to-MIDI piano-roll:
    - Cache key: (file_path, start_sample, end_sample, sample_rate, n_fft, hop_length)
    - `frequencies` and `magnitudes` are ignored in the key but used directly if provided.
    """
    def _calc(f):
        if f <= 0.0:
            return 0
        return int(round(69 + 12 * math.log2(f / 440.0)))

    vec = np.vectorize(_calc, otypes=[int])
    midi = vec(frequencies)

    n_bins, n_frames = midi.shape
    piano_roll = np.zeros((128, n_frames), dtype=float)
    for note in range(128):
        mask = (midi == note)
        masked = magnitudes * mask
        piano_roll[note, :] = masked.max(axis=0)

    return piano_roll


class FrequencyToMidi:
    def __init__(
            self,
            logger: HoornLogger,
    ):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._logger.trace(f"Initialized successfully", separator=self._separator)

    def convert(
            self,
            *,
            file_path: Path,
            sample_rate: int,
            frequencies: np.ndarray = None,
            magnitudes: np.ndarray = None,
    ) -> np.ndarray:
        self._logger.debug(f"Converting to MIDI for {file_path.name}", separator=self._separator)
        piano_roll = _convert_to_midi(
            file_path=file_path,
            sample_rate=sample_rate,
            frequencies=frequencies,
            magnitudes=magnitudes,
        )
        self._logger.debug(
            f"Piano roll shape: {piano_roll.shape}",
            separator=self._separator
        )
        return piano_roll
