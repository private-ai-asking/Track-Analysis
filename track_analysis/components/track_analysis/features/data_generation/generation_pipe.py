from typing import List

from track_analysis.components.md_common_python.py_common.multithreading.thread_manager import ThreadManagerConfig, \
    ThreadManager
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.contexts import DataGenerationPipeContext, \
    DataGenerationPipeConfiguration, BatchContext
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


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
