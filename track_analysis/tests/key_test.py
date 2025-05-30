from pathlib import Path
from typing import Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.key_progression_analyzer import \
    KeyProgressionAnalyzer


class KeyProgressionTest(TestInterface):
    def __init__(self, logger: HoornLogger, modulation_penalty: float = 6.0):
        super().__init__(logger, is_child=True)
        self._separator = 'KeyProgressionTest'
        self._analyzer = KeyProgressionAnalyzer(logger, modulation_penalty)

    def test(self, file_path: Path, time_signature: Tuple[int,int] = (4,4), segment_beat_level: int = 3) -> None:
        runs = self._analyzer.analyze(file_path, time_signature, segment_beat_level)
        for run in runs:
            duration = run.end_time - run.start_time
            self._logger.info(
                f"[segment {run.start_index}] {self._format_time(run.start_time)} -> {self._format_time(run.end_time)}"
                f" ({duration:.2f}s) => {run.state_label}",
                separator=self._separator
            )
        self._logger.info("Key progression complete.", separator=self._separator)

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
