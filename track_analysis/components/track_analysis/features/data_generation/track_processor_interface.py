import abc
from abc import abstractmethod

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.model.audio_info import AudioInfo


class ITrackProcessorStrategy(abc.ABC):
    def __init__(self, logger: HoornLogger, specific_processor: str, is_child: bool=False):
        self._separator = f"DataGenerator.TrackProcessor-{specific_processor}"
        self._logger = logger
        self._is_child = is_child

        if not self._is_child:
            raise NotImplementedError("You cannot directly instantiate an interface.")

        self._logger.trace("Successfully initialized.", separator=self._separator)

    @abstractmethod
    def process_track(self, track: AudioInfo) -> AudioInfo:
        """Processes a track according to the implementation."""
        if not self._is_child:
            raise NotImplementedError("Subclasses should implement this method.")
