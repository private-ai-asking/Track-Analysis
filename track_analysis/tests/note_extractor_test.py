from pathlib import Path
from typing import Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.note_extractor import \
    NoteExtractor


class NoteExtractorTest:
    def __init__(self, logger: HoornLogger) -> None:
        self._logger: HoornLogger = logger
        self._separator: str = "NoteExtractorTest"
        self._logger.trace("Successfully initialized.", separator=self._separator)
        self._note_extractor: NoteExtractor = NoteExtractor(logger, hop_length_ms=512, subdivisions_per_beat=2)

    def test(self,
             file_path: Path,
             time_signature: Tuple[int,int] = (4,4),
             segment_beat_level: int = 3,
             ) -> None:
        self._logger.info(f"Testing note extraction with segment beat level of {segment_beat_level}.", separator=self._separator)
        self._note_extractor.extract(file_path, time_signature, segment_beat_level)

