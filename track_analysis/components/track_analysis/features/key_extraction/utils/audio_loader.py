import os
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    return librosa.load(path, sr=None)

class AudioLoader:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator: str = "AudioLoader"

        cache_dir = EXPENSIVE_CACHE_DIRECTORY / "audio loading"
        os.makedirs(cache_dir, exist_ok=True)
        self._compute = Memory(cache_dir, verbose=0).cache(_load_audio)


    def load(self, path: Path) -> tuple[np.ndarray,int]:
        audio, sr = self._compute(path)
        self._logger.debug(f"Loaded {len(audio)} samples @ {sr}Hz", separator=self._separator)
        return audio, sr
