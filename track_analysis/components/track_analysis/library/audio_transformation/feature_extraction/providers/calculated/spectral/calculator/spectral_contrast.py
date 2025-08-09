from pathlib import Path

import numpy as np
from librosa.feature import spectral_contrast

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["audio"])
def compute_spectral_contrast(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        n_fft:         int = 2048,
        n_bands:       int = 6,
        audio:         np.ndarray = None,
) -> TimedCacheResult[np.ndarray]:
    """
    Returns spectral contrast for the given range.
    """
    audio = audio[start_sample:end_sample]

    # noinspection PyTypeChecker
    return spectral_contrast(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_bands=n_bands
    )


class SpectralContrastExtractor:
    def __init__(self, logger: HoornLogger):
        self._logger    = logger
        self._separator = self.__class__.__name__

    def extract(
            self,
            file_path:     Path,
            start_sample:  int,
            end_sample:    int,
            sample_rate:   int,
            hop_length:    int,
            n_fft:         int = 2048,
            n_bands:       int = 6,
            audio:         np.ndarray = None,
    ) -> TimedCacheResult[np.ndarray]:
        """
        Extracts spectral contrast envelope, cached.
        """
        self._logger.debug(
            f"SpectralContrastExtractor: {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return compute_spectral_contrast(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_bands=n_bands,
            audio=audio,
        )
