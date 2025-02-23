import threading
from typing import List, Any

import pydantic

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.header import Header


class BatchContext(pydantic.BaseModel):
    tracks_to_process_number: int # Already thread-safe, because doesn't change.

    processed_tracks_number: int
    processed_tracks: List[AudioInfo]

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        self._processed_track_number_lock: threading.Lock = threading.Lock()
        self._processed_tracks_lock: threading.Lock = threading.Lock()

    def get_processed_tracks_number_thread_safe(self) -> int:
        with self._processed_track_number_lock:
            return self.processed_tracks_number

    def increase_processed_tracks_number_thread_safe(self, n: int) -> None:
        with self._processed_track_number_lock:
            self.processed_tracks_number += n

    def get_processed_tracks_thread_safe(self) -> List[AudioInfo]:
        with self._processed_tracks_lock:
            return self.processed_tracks

    def extend_processed_tracks_threadsafe(self, to_extend: List[AudioInfo]) -> None:
        with self._processed_tracks_lock:
            self.processed_tracks.extend(to_extend)

class DataGenerationPipeConfiguration(pydantic.BaseModel):
    headers_to_fill: List[Header]
    batch_size: int

class DataGenerationPipeContext(pydantic.BaseModel):
    logger: HoornLogger
    audio_file_handler: AudioFileHandler
    audio_calculator: AudioCalculator
    time_utils: TimeUtils

    model_config = {
        "arbitrary_types_allowed": True
    }
