from pathlib import Path

import numpy as np
from librosa.feature import spectral_contrast

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.cacheing.shared import MEMORY


@MEMORY.cache(ignore=["audio"])
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
) -> np.ndarray:
    """
    Returns spectral contrast for the given range.
    """
    if audio is None:
        audio = np.memmap(
            str(file_path),
            dtype="float32",
            mode="r",
            offset=start_sample * 4,
            shape=(end_sample - start_sample,),
        )
    else:
        audio = audio[start_sample:end_sample]

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
    ) -> np.ndarray:
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
