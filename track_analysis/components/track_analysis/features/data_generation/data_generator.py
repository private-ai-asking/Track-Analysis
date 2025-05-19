from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.md_common_python.py_common.time_handling.time_utils import time_operation
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.constants import ROOT_MUSIC_LIBRARY, OUTPUT_DIRECTORY
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.data_generation.generation_pipeline import DataGenerationPipeline
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import PipelineContextModel


class DataGenerator:
    """Used for filling in the missing data for all tracks. To add the new header/values backwardly."""
    def __init__(self,
                 logger: HoornLogger,
                 audio_file_handler: AudioFileHandler,
                 audio_calculator: AudioCalculator,
                 time_utils: TimeUtils,
                 string_utils: StringUtils):
        self._separator = "DataGenerator"

        self._logger = logger
        self._audio_file_handler = audio_file_handler
        self._audio_calculator = audio_calculator
        self._time_utils = time_utils
        self._string_utils = string_utils

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def _flow_pipeline(self, pipeline: AbPipeline, data: PipelineContextModel):
        @time_operation(logger=self._logger, time_utils=self._time_utils, separator=self._separator)
        def flow_wrapper():
            pipeline.flow(data)

        flow_wrapper()

    def generate_data(self, headers: List[Header], batch_size: int=64):
        context = PipelineContextModel(
            source_dir=ROOT_MUSIC_LIBRARY,
            main_data_output_file_path=OUTPUT_DIRECTORY.joinpath("data.csv"),
        )

        self._logger.trace("Generating data.", separator=self._separator)

        pipeline = DataGenerationPipeline(self._logger, headers, self._audio_file_handler, self._audio_calculator, self._time_utils, self._string_utils, batch_size=batch_size)
        pipeline.build_pipeline()
        self._flow_pipeline(pipeline, context)
