import os
import pprint
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

def _extract(spectral_data: np.ndarray, sample_rate: int, min_frq: int, max_frq: int, hop_length: int, n_fft: int) -> Tuple[np.ndarray, np.ndarray]:
    return librosa.piptrack(
        S=spectral_data,
        sr=sample_rate,
        fmin=min_frq,
        fmax=max_frq,
        hop_length=hop_length,
        n_fft=n_fft,
    )


class SpectralPeakExtractor:
    def __init__(self, logger: HoornLogger, cache_dir: Path, min_frequency_hz: int, max_frequency_hz: int, hop_length_samples: int, n_fft: int):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._min_frequency_hz = min_frequency_hz
        self._max_frequency_hz = max_frequency_hz

        self._hop_length_samples = hop_length_samples
        self._n_fft = n_fft

        os.makedirs(cache_dir, exist_ok=True)
        self._extract = Memory(cache_dir, verbose=0, compress=3).cache(_extract)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def extract_spectral_peaks(self, spectral_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        self._logger.debug(f"Extracting spectral peaks with config: "
                           f"(sr={sample_rate};"
                           f"min_freq={self._min_frequency_hz}Hz;"
                           f"max_freq={self._max_frequency_hz}Hz;"
                           f"hop_length={self._hop_length_samples};"
                           f"n_fft={self._n_fft})", separator=self._separator)

        pitches, magnitudes = self._extract(spectral_data, sample_rate, self._min_frequency_hz, self._max_frequency_hz, self._hop_length_samples, self._n_fft)

        self._logger.debug(f"Pitches and Magnitudes shape: {pitches.shape} | {magnitudes.shape}", separator=self._separator)
        self._logger.debug(f"Pitches:\n{pprint.pformat(pitches)}", separator=self._separator)
        self._logger.debug(f"Magnitudes:\n{pprint.pformat(magnitudes)}", separator=self._separator)

        return pitches, magnitudes
