from pathlib import Path
from typing import Tuple

import librosa
import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.audio_segmenter import AudioSegmenter


class SegmentationTest:
    def __init__(self, logger: HoornLogger, min_segment_level: int = 3, subdivisions_per_beat: int = 2):
        self._logger = logger
        self._separator: str = "SegmentationTest"
        self._min_segment_level = min_segment_level
        # number of equal subdivisions of each beat, to generate a fixed 4-level hierarchy
        self._subdivisions_per_beat = subdivisions_per_beat
        self._audio_segmenter: AudioSegmenter = AudioSegmenter(logger, subdivisions_per_beat)
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def test(self, file_path: Path, time_signature: Tuple[int, int] = (4, 4)):
        audio_samples, sample_rate = self._load_track(file_path)
        result = self._audio_segmenter.get_segments(audio_samples, sample_rate, time_signature)
        start_times = result.start_times
        durations = result.durations

        for idx, (st, dur) in enumerate(zip(start_times, durations), start=1):
            self._logger.info(
                f"{self._format_time(st)}: segment {idx} (duration={dur:.3f}s)",
                separator=self._separator
            )

    def _load_track(self, file_path: Path) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(file_path, sr=None)
        self._logger.debug(
            f"Loaded {len(audio)} frames @ {sr}Hz",
            separator=self._separator
        )
        return audio, sr

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
