import os
import pprint
from pathlib import Path

import librosa
import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

def _extract(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))


class MagnitudeSpectogramExtractor:
    def __init__(self, logger: HoornLogger, cache_dir: Path, n_fft: int = 2048, hop_length: int = 512):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._logger.trace("Successfully initialized.", separator=self._separator)

        self._n_fft = n_fft
        self._hop_length = hop_length

        os.makedirs(cache_dir, exist_ok=True)
        self._extract = Memory(cache_dir, verbose=0).cache(_extract)


    def extract_magnitude_spectogram(self, harmonic_audio: np.ndarray) -> np.ndarray:
        extracted_magnitude_spectogram = self._extract(harmonic_audio, self._n_fft, self._hop_length)
        self._logger.debug(f"Extracted Magnitude Spectogram Shape: {extracted_magnitude_spectogram.shape}", separator=self._separator)
        self._logger.debug(f"Extracted Magnitude Spectogram:\n{pprint.pformat(extracted_magnitude_spectogram)}", separator=self._separator)

        return extracted_magnitude_spectogram
