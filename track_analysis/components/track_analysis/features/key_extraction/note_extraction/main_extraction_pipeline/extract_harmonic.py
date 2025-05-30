import os
import pprint
from pathlib import Path
from typing import Tuple

import librosa.effects
import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


def _extract_harmonic(raw_audio: np.ndarray, kernel_size: int, hop_length: int = 512, n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    harmonic, percussive = librosa.effects.hpss(
        y=raw_audio,
        kernel_size=kernel_size,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=None,
        margin=1,
        power=2
    )
    return harmonic, percussive

class HarmonicExtractor:
    def __init__(self, logger: HoornLogger, cache_dir: Path, hop_length_samples: int = 512, n_fft: int = 2048):
        self._logger = logger
        self._separator: str = self.__class__.__name__

        self._hop_length_samples = hop_length_samples
        self._n_fft = n_fft

        os.makedirs(cache_dir, exist_ok=True)
        self._compute = Memory(cache_dir, verbose=0).cache(_extract_harmonic)

        self._logger.trace(f"Successfully initialized.", separator=self._separator)

    def extract_harmonic(self, audio: np.ndarray,sample_rate: int, tempo_bpm: float) -> Tuple[np.ndarray, np.ndarray]:
        frames_per_beat: float = sample_rate * 60 / tempo_bpm / self._hop_length_samples
        kernel_size: int = int(round(frames_per_beat))

        self._logger.debug(f"Extracting harmonic using params: "
                           f"(frames_per_beat={frames_per_beat:.4f};"
                           f"kernel_size={kernel_size};"
                           f"hop_length={self._hop_length_samples};"
                           f"n_fft={self._n_fft}).", separator=self._separator)

        harmonic, percussive = self._compute(audio, kernel_size, self._hop_length_samples, self._n_fft)

        self._logger.debug(f"Harmonic & Percussive Shapes: {harmonic.shape} | {percussive.shape}.", separator=self._separator)
        self._logger.debug(f"Harmonic:\n{pprint.pformat(harmonic)}", separator=self._separator)
        self._logger.debug(f"Percussive:\n{pprint.pformat(percussive)}", separator=self._separator)

        return harmonic, percussive

