from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel, AudioFileHandler
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import LibraryDataGenerationPipelineContext


class GetStreamInfo(IPipe):
    def __init__(self, logger: HoornLogger, audio_file_handler: AudioFileHandler):
        self._separator = "BuildCSV.GetStreamInfo"
        self._audio_file_handler = audio_file_handler

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        stream_infos: List[AudioStreamsInfoModel] = self._audio_file_handler.get_audio_streams_info_batch(data.filtered_audio_file_paths)
        data.extracted_stream_info = stream_infos
        return data
