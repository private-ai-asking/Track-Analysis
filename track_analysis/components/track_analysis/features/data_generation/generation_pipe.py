from typing import List, Dict, Union

from track_analysis.components.md_common_python.py_common.multithreading.thread_manager import ThreadManagerConfig, \
    ThreadManager
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.contexts import DataGenerationPipeContext, \
    DataGenerationPipeConfiguration, BatchContext
from track_analysis.components.track_analysis.features.data_generation.track_processor_interface import \
    ITrackProcessorStrategy
from track_analysis.components.track_analysis.features.data_generation.track_processors.true_peak_processor import \
    TruePeakTrackProcessor
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
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

    def __get_track_processor(self, header: Header) -> Union[ITrackProcessorStrategy, None]:
        processor_mapping: Dict[Header, ITrackProcessorStrategy] = {
            Header.True_Peak: TruePeakTrackProcessor(self._logger, self._audio_file_handler, self._audio_calculator)
        }

        processor = processor_mapping.get(header, None)

        if processor is None:
            self._logger.warning(f"No processor found for header - unsupported: {header}.", separator=self._separator)

        return processor

    def __handle_batch(self, batch: List[AudioInfo], worker_context: BatchContext):
        batch_results: List[AudioInfo] = []

        for track in batch:
            batch_results.append(worker_context.associated_track_processor.process_track(track))

            worker_context.increase_processed_tracks_number_thread_safe(1)
            processed_tracks = worker_context.get_processed_tracks_number_thread_safe()

            self._logger.info(
                f"Processed {processed_tracks}/{worker_context.tracks_to_process_number} ({round(processed_tracks / worker_context.tracks_to_process_number * 100, 4)}%) tracks.",
                separator=self._separator
            )

        worker_context.extend_processed_tracks_threadsafe(batch_results)

    def flow(self, context: PipelineContextModel) -> PipelineContextModel:
        for header in self._configuration.headers_to_fill:
            track_processor = self.__get_track_processor(header)
            if track_processor is None:
                continue
            else:
                cache: List[AudioInfo] = context.loaded_audio_info_cache

                worker_context: BatchContext = BatchContext(
                    tracks_to_process_number = len(cache),
                    processed_tracks_number = 0,
                    processed_tracks  = [],
                    associated_track_processor = track_processor,
                )

                # Split the work into batches
                batches: List[List[AudioInfo]] = []
                for i in range(0, worker_context.tracks_to_process_number, self._configuration.batch_size):
                    batches.append(cache[i : i + self._configuration.batch_size])

                self._thread_manager.work_batches(batches, worker_context)

                updated_rows = worker_context.processed_tracks

                context.loaded_audio_info_cache = updated_rows
        return context
