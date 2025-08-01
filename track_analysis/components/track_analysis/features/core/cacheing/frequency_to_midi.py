import math
from pathlib import Path
import numpy as np
import librosa

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.cacheing.shared import MEMORY


@MEMORY.cache(ignore=["frequencies", "magnitudes"])
def _convert_to_midi(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        frequencies: np.ndarray = None,
        magnitudes: np.ndarray = None,
) -> np.ndarray:
    """
    Cached frequency-to-MIDI piano-roll:
    - Cache key: (file_path, start_sample, end_sample, sample_rate, n_fft, hop_length)
    - `frequencies` and `magnitudes` are ignored in the key but used directly if provided.
    """
    # If not provided, recompute frequencies & magnitudes for the segment
    if frequencies is None or magnitudes is None:
        length = end_sample - start_sample
        audio = np.memmap(
            str(file_path), dtype="float32", mode="r",
            offset=start_sample * 4,
            shape=(length,)
        )
        S = np.abs(
            librosa.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length
            )
        )
        frequencies, magnitudes = librosa.piptrack(
            S=S,
            sr=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft
        )

    # Map each frequency to its nearest MIDI note
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
            hop_length: int,
            n_fft: int
    ):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._hop_length = hop_length
        self._n_fft = n_fft
        self._logger.trace(
            f"Initialized with n_fft={n_fft}, hop_length={hop_length}",
            separator=self._separator
        )

    def convert(
            self,
            *,
            file_path: Path,
            start_sample: int,
            end_sample: int,
            sample_rate: int,
            frequencies: np.ndarray = None,
            magnitudes: np.ndarray = None,
    ) -> np.ndarray:
        """
        Convert a segment into a MIDI piano-roll, caching on file + range + STFT params.
        If `frequencies` and `magnitudes` are provided, they are used directly;
        otherwise they are recomputed from the file.
        """
        self._logger.debug(
            f"Converting to MIDI for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator
        )
        piano_roll = _convert_to_midi(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            frequencies=frequencies,
            magnitudes=magnitudes,
        )
        self._logger.debug(
            f"Piano roll shape: {piano_roll.shape}",
            separator=self._separator
        )
        return piano_roll
