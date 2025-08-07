from pathlib import Path
from typing import Tuple

import numpy as np
import librosa

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["spectral_data"])
def _compute_spectral_peaks(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        min_frq: int,
        max_frq: int,
        hop_length: int,
        n_fft: int,
        spectral_data: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cached spectral peak extraction via librosa.piptrack:
    - Cache key: (file_path, start_sample, end_sample, sample_rate, min_frq, max_frq, hop_length, n_fft)
    - `spectral_data` is ignored in the cache key but used if provided.
    """
    spectrogram = spectral_data

    # Extract pitches and magnitudes
    pitches, magnitudes = librosa.piptrack(
        S=spectrogram,
        sr=sample_rate,
        fmin=min_frq,
        fmax=max_frq,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    return pitches, magnitudes


class SpectralPeakExtractor:
    def __init__(
            self,
            logger: HoornLogger,
            min_frequency_hz: int,
            max_frequency_hz: int,
            hop_length_samples: int,
            n_fft: int,
    ):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._min_frequency_hz = min_frequency_hz
        self._max_frequency_hz = max_frequency_hz
        self._hop_length_samples = hop_length_samples
        self._n_fft = n_fft
        self._logger.trace(
            f"Initialized SpectralPeakExtractor with min_freq={min_frequency_hz}, max_freq={max_frequency_hz}, hop_length={hop_length_samples}, n_fft={n_fft}",
            separator=self._separator
        )

    def extract_spectral_peaks(
            self,
            *,
            file_path: Path,
            start_sample: int,
            end_sample: int,
            sample_rate: int,
            spectral_data: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._logger.debug(
            f"Extracting spectral peaks for {file_path.name}[{start_sample}:{end_sample}] with freq_range=({self._min_frequency_hz}-{self._max_frequency_hz})Hz, hop_length={self._hop_length_samples}, n_fft={self._n_fft}",
            separator=self._separator
        )
        pitches, magnitudes = _compute_spectral_peaks(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            min_frq=self._min_frequency_hz,
            max_frq=self._max_frequency_hz,
            hop_length=self._hop_length_samples,
            n_fft=self._n_fft,
            spectral_data=spectral_data,
        )
        self._logger.debug(
            f"Pitches shape={pitches.shape}, Magnitudes shape={magnitudes.shape}",
            separator=self._separator
        )
        return pitches, magnitudes
