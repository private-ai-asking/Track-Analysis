from pathlib import Path

import numpy as np
from librosa.feature import spectral_flatness

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.caching.cached_operations.shared import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def compute_spectral_flatness(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        n_fft:         int = 2048,
        audio:         np.ndarray = None,
) -> np.ndarray:
    """
    Returns spectral flatness for the given range.
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

    return spectral_flatness(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length
    )


class SpectralFlatnessExtractor:
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
            audio:         np.ndarray = None,
    ) -> np.ndarray:
        """
        Extracts spectral flatness envelope, cached.
        """
        self._logger.debug(
            f"SpectralFlatnessExtractor: {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return compute_spectral_flatness(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            audio=audio,
        )
