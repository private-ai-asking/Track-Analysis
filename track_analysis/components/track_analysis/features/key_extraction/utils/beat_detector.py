import os
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY


def _perform_beat_track(audio: np.ndarray, sample_rate: int):
    tempo, frames = librosa.beat.beat_track(
        y=audio, sr=sample_rate, units='frames', trim=False
    )
    return tempo, frames


class BeatDetector:
    """
    Detects beats and estimates tempo from audio.
    """
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

        cache_dir = CACHE_DIRECTORY / "beat detection"
        os.makedirs(cache_dir, exist_ok=True)

        self._compute = Memory(cache_dir, verbose=0).cache(_perform_beat_track)

    def detect(
            self,
            audio: np.ndarray,
            sample_rate: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        tempo, frames = self._compute(audio, sample_rate)
        tempo = tempo[0]
        times = librosa.frames_to_time(frames, sr=sample_rate)

        self._logger.info(
            f"Estimated tempo: {tempo:.2f} BPM",
            separator=self._separator,
        )
        self._logger.debug(
            f"Detected {len(frames)} beats.",
            separator=self._separator,
        )
        return tempo, frames, times
