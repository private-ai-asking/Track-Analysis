from typing import Tuple

import librosa
import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class BeatDetector:
    """
    Detects beats and estimates tempo from audio.
    """
    def __init__(self, logger: HoornLogger, separator: str):
        self._logger = logger
        self._separator = separator

    def detect(
            self,
            audio: np.ndarray,
            sample_rate: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        tempo, frames = librosa.beat.beat_track(
            y=audio, sr=sample_rate, units='frames', trim=False
        )
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
