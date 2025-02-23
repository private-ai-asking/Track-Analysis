import threading
from typing import List, Any

import pydantic

from track_analysis.components.md_common_python.py_common.multithreading.thread_manager import ThreadManagerConfig, \
    ThreadManager
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.track_analysis.constants import ROOT_MUSIC_LIBRARY, OUTPUT_DIRECTORY
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel, AudioFileHandler
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel
from track_analysis.components.track_analysis.pipeline.pipes.load_cache import LoadCache
from track_analysis.components.track_analysis.pipeline.pipes.make_csv import MakeCSV


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

    model_config = {
        "arbitrary_types_allowed": True
    }

class DataGenerator:
    """Used for filling in the missing data for all tracks. To add the new header/values backwardly."""
    def __init__(self, logger: HoornLogger, audio_file_handler: AudioFileHandler, audio_calculator: AudioCalculator):
        self._separator = "DataGenerator"

        self._logger = logger
        self._audio_file_handler = audio_file_handler
        self._audio_calculator = audio_calculator

        self._logger.trace("Successfully initialized.", separator=self._separator)

    class DataGenerationPipe(IPipe):
        def __init__(self, context: DataGenerationPipeContext, configuration: DataGenerationPipeConfiguration):
            self._separator = "DataGenerator.DataGenerationPipe"
            self._logger = context.logger
            self._audio_file_handler = context.audio_file_handler
            self._audio_calculator = context.audio_calculator
            self._configuration = configuration

            thread_config: ThreadManagerConfig = ThreadManagerConfig(
                num_threads=8,
                worker_template=self.__handle_batch,
                worker_name="DataGenerator.DataGenerationPipe.BatchHandler"
            )

            self._thread_manager: ThreadManager = ThreadManager(self._logger, thread_config)
            self._logger.trace("Successfully initialized.", separator=self._separator)

        def __process_track(self, cached_track_info: AudioInfo) -> AudioInfo:
            """Processes a single track and returns the updated AudioInfo."""
            file_info: AudioStreamsInfoModel = self._audio_file_handler.get_audio_streams_info(cached_track_info.path)
            true_peak = self._audio_calculator.calculate_true_peak(file_info.sample_rate_Hz, file_info.samples_librosa)

            updated_metadata = []

            for metadata_item_original in cached_track_info.metadata:
                updated_metadata.append(metadata_item_original.model_copy(deep=True))

            updated_metadata.append(AudioMetadataItem(header=Header.True_Peak, description="", value=true_peak))
            return AudioInfo(path=cached_track_info.path, metadata=updated_metadata, timeseries_data=cached_track_info.timeseries_data)

        def __handle_batch(self, batch: List[AudioInfo], worker_context: BatchContext):
            batch_results: List[AudioInfo] = []

            for track in batch:
                batch_results.append(self.__process_track(track))
                
                worker_context.increase_processed_tracks_number_thread_safe(1)
                processed_tracks = worker_context.get_processed_tracks_number_thread_safe()

                self._logger.info(
                    f"Processed {processed_tracks}/{worker_context.tracks_to_process_number} ({round(processed_tracks / worker_context.tracks_to_process_number * 100, 4)}%) tracks.",
                    separator=self._separator
                )

            worker_context.extend_processed_tracks_threadsafe(batch_results)

        def flow(self, context: PipelineContextModel) -> PipelineContextModel:
            cache: List[AudioInfo] = context.loaded_audio_info_cache

            worker_context: BatchContext = BatchContext(
                tracks_to_process_number = len(cache),
                processed_tracks_number = 0,
                processed_tracks  = []
            )

            # Split the work into batches
            batches: List[List[AudioInfo]] = []
            for i in range(0, worker_context.tracks_to_process_number, self._configuration.batch_size):
                batches.append(cache[i : i + self._configuration.batch_size])

            self._thread_manager.work_batches(batches, worker_context)

            updated_rows = worker_context.processed_tracks

            context.loaded_audio_info_cache = updated_rows
            return context

    class DataGenerationPipeline(AbPipeline):
        def __init__(self,
                     logger: HoornLogger,
                     headers_to_fill: List[Header],
                     audio_file_handler: AudioFileHandler,
                     audio_calculator: AudioCalculator,
                     batch_size: int=64):
            self._logger = logger
            self._headers_to_fill = headers_to_fill

            self._data_generation_pipe_context: DataGenerationPipeContext = DataGenerationPipeContext(
                logger=logger,
                audio_file_handler=audio_file_handler,
                audio_calculator=audio_calculator
            )

            self._data_generation_pipe_configuration: DataGenerationPipeConfiguration = DataGenerationPipeConfiguration(
                headers_to_fill=headers_to_fill,
                batch_size=batch_size
            )

            super().__init__()

        def build_pipeline(self):
            self._add_step(LoadCache(self._logger))
            self._add_step(DataGenerator.DataGenerationPipe(self._data_generation_pipe_context, self._data_generation_pipe_configuration))
            self._add_step(MakeCSV(self._logger))

    def generate_data(self, headers: List[Header], batch_size: int=64):
        context = PipelineContextModel(
            source_dir=ROOT_MUSIC_LIBRARY,
            main_data_output_file_path=OUTPUT_DIRECTORY.joinpath("data.csv"),
        )

        pipeline = DataGenerator.DataGenerationPipeline(self._logger, headers, self._audio_file_handler, self._audio_calculator, batch_size)
        pipeline.build_pipeline()
        pipeline.flow(context)
