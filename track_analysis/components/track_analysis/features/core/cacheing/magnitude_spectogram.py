from pathlib import Path
import numpy as np
import librosa

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.cacheing.shared import MEMORY


@MEMORY.cache(ignore=["audio"])
def _compute_magnitude_spectrogram(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        audio: np.ndarray = None,
) -> np.ndarray:
    """
    Cached magnitude spectrogram computation:
    - Cache key: (file_path, start_sample, end_sample, n_fft, hop_length)
    - `audio` is ignored in the cache key but used if provided.
    """
    # Obtain the audio segment, either via memmap or slicing
    if audio is None:
        length = end_sample - start_sample
        segment = np.memmap(
            str(file_path), dtype="float32", mode="r",
            offset=start_sample * 4,
            shape=(length,)
        )
    else:
        segment = audio[start_sample:end_sample]

    # Compute magnitude spectrogram
    spectrogram = np.abs(
        librosa.stft(
            segment,
            n_fft=n_fft,
            hop_length=hop_length
        )
    )
    return spectrogram


class MagnitudeSpectrogramExtractor:
    def __init__(
            self,
            logger: HoornLogger,
            n_fft: int = 2048,
            hop_length: int = 512
    ):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._logger.trace(
            f"Initialized with n_fft={n_fft}, hop_length={hop_length}",
            separator=self._separator
        )

    def extract(
            self,
            *,
            file_path: Path,
            start_sample: int,
            end_sample: int,
            audio: np.ndarray = None
    ) -> np.ndarray:
        """
        Extracts (and caches) the magnitude spectrogram for the specified segment.

        If `audio` is provided, uses that array slice; otherwise, memory-maps the file.
        """
        self._logger.debug(
            f"Computing magnitude spectrogram for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator
        )
        spectrogram = _compute_magnitude_spectrogram(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            audio=audio
        )
        self._logger.debug(
            f"Spectrogram shape: {spectrogram.shape}",
            separator=self._separator
        )
        return spectrogram
