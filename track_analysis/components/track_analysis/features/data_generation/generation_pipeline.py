from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.data_generation.contexts import DataGenerationPipeContext, \
    DataGenerationPipeConfiguration
from track_analysis.components.track_analysis.features.data_generation.generation_pipe import DataGenerationPipe
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.pipes.load_cache import LoadCache
from track_analysis.components.track_analysis.pipeline.pipes.make_csv import MakeCSV


class DataGenerationPipeline(AbPipeline):
    def __init__(self,
                 logger: HoornLogger,
                 headers_to_fill: List[Header],
                 audio_file_handler: AudioFileHandler,
                 audio_calculator: AudioCalculator,
                 time_utils: TimeUtils,
                 batch_size: int=64):
        self._logger = logger
        self._headers_to_fill = headers_to_fill
        self._time_utils = time_utils

        self._data_generation_pipe_context: DataGenerationPipeContext = DataGenerationPipeContext(
            logger=logger,
            audio_file_handler=audio_file_handler,
            audio_calculator=audio_calculator,
            time_utils=self._time_utils
        )

        self._data_generation_pipe_configuration: DataGenerationPipeConfiguration = DataGenerationPipeConfiguration(
            headers_to_fill=headers_to_fill,
            batch_size=batch_size
        )

        super().__init__()

    def build_pipeline(self):
        self._add_step(LoadCache(self._logger))
        self._add_step(DataGenerationPipe(self._data_generation_pipe_context, self._data_generation_pipe_configuration))
        self._add_step(MakeCSV(self._logger))
