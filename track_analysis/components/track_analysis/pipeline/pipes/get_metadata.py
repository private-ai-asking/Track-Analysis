from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class GetAudioMetadata(IPipe):
    def __init__(self, logger: HoornLogger, tag_extractor: TagExtractor):
        self._separator = "GetAudioMetadataPipe"

        self._logger = logger
        self._tag_extractor = tag_extractor
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Extracting all audio metadata...", separator=self._separator)
        audio_info: List[AudioInfo] = []

        for track_path in data.audio_file_paths:
            audio_info.append(self._tag_extractor.extract(track_path))
            self._logger.trace(f"Extracted audio metadata for: {track_path}", separator=self._separator)
            self._logger.debug(f"Extracted audio metadata: {audio_info}", separator=self._separator)

        data.audio_info = audio_info

        self._logger.trace("Successfully extracted all audio metadata.", separator=self._separator)
        return data
