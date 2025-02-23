import threading
from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.track_analysis.constants import ROOT_MUSIC_LIBRARY, OUTPUT_DIRECTORY
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.build_csv_pipeline import BuildCSVPipeline

from track_analysis.components.track_analysis.pipeline.locate_paths_pipeline import LocatePathsPipeline
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel
from track_analysis.components.track_analysis.pipeline.pipes.load_cache import LoadCache
from track_analysis.components.track_analysis.pipeline.pipes.make_csv import MakeCSV


class App:
    def __init__(self, logger: HoornLogger):
        self._user_input_helper: UserInputHelper = UserInputHelper(logger)
        self._tag_extractor: TagExtractor = TagExtractor(logger)
        self._file_handler: FileHandler = FileHandler()
        self._audio_file_handler: AudioFileHandler = AudioFileHandler(logger)
        self._audio_calculator: AudioCalculator = AudioCalculator(logger)
        self._logger = logger

    def run(self):
        cmd: CommandLineInterface = CommandLineInterface(self._logger)
        cmd.add_command(["extract_tags_debug", "etd"], "Debugs the extract tags function.", self._debug_extract_tags)
        cmd.add_command(["make_csv", "mc"], "Makes a CSV file from the extracted metadata.", self._make_csv)
        cmd.add_command(["add_path_to_metadata", "apm"], "Adds the path of a file to the metadata.", self._add_path_to_metadata)
        cmd.add_command(["generate_new_data", "gnd"], "Fills in the newly added header(s) since last cache update.", self._generate_new_data)
        cmd.start_listen_loop()

    def _generate_new_data(self):
        context = PipelineContextModel(
            source_dir=ROOT_MUSIC_LIBRARY,
            output_file_path=OUTPUT_DIRECTORY.joinpath("data.csv"),
        )

        context = LoadCache(self._logger).flow(context)

        cache: List[AudioInfo] = context.loaded_audio_info_cache

        to_process = len(cache)
        processed: int = 0
        batch_size = 64
        num_threads = 8
        results_queue = []
        lock = threading.Lock()
        progress_lock = threading.Lock()
        semaphore = threading.Semaphore(num_threads)

        def process_track(cached_track_info: AudioInfo) -> AudioInfo:
            """Processes a single track and returns the updated AudioInfo."""
            file_info: AudioStreamsInfoModel = self._audio_file_handler.get_audio_streams_info(cached_track_info.path)
            true_peak = self._audio_calculator.calculate_true_peak(file_info.sample_rate_Hz, file_info.samples_librosa)
            updated_metadata = []

            for metadata_item_original in cached_track_info.metadata:
                updated_metadata.append(metadata_item_original.model_copy(deep=True))

            updated_metadata.append(AudioMetadataItem(header=Header.True_Peak, description="", value=true_peak))
            return AudioInfo(path=cached_track_info.path, metadata=updated_metadata)

        def worker(batch: List[AudioInfo]):
            """Processes a batch of tracks, logging progress for each."""
            try:
                semaphore.acquire()
                nonlocal processed
                batch_results = []
                for cached_track_info in batch:
                    updated_track_info = process_track(cached_track_info)  # Process single track

                    with lock:
                        batch_results.append(updated_track_info)

                    with progress_lock:
                        processed += 1
                        self._logger.info(
                            f"Processed {processed}/{to_process} ({round(processed / to_process * 100, 4)}%) tracks."
                        )

                with lock:
                    results_queue.extend(batch_results)
            finally:
                semaphore.release()

        # Split the work into batches
        batches: List[List[AudioInfo]] = []
        for i in range(0, to_process, batch_size):
            batches.append(cache[i : i + batch_size])

        # Use threads to process batches
        threads = []
        for batch in batches:
            thread = threading.Thread(target=worker, args=(batch,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        updated_rows = results_queue

        context.loaded_audio_info_cache = updated_rows

        MakeCSV(self._logger).flow(context)

    def _debug_extract_tags(self):
        def _always_true_validator(_: str) -> (bool, str):
            return True, ""

        # W:\media\music\[02] organized\[01] hq\Classical\Ludovico Einaudi\Elegy For The Arctic\01 Elegy for the Arctic.flac
        # W:\media\music\[02] organized\[02] lq\CCM\Champion\15 - Beckah Shae - Incorruptible (David Thulin remix).flac
        file_to_check: str = self._user_input_helper.get_user_input("Please enter the path to the audio file you want to extract:", str, validator_func=_always_true_validator)

        result: AudioInfo =self._tag_extractor.extract(Path(file_to_check))

        for metadata_item in result.metadata:
            self._logger.info(f"{metadata_item.header} - {metadata_item.description}: {metadata_item.value}")

    def _add_path_to_metadata(self):
        pipeline_context = PipelineContextModel(
            source_dir=ROOT_MUSIC_LIBRARY,
            output_file_path=OUTPUT_DIRECTORY.joinpath("data.csv"),
        )

        pipeline = LocatePathsPipeline(self._logger, self._file_handler, self._tag_extractor)
        pipeline.build_pipeline()
        pipeline.flow(pipeline_context)

        self._logger.info("Paths have been successfully matched.")

    def _make_csv(self):
        pipeline_context = PipelineContextModel(
            source_dir=ROOT_MUSIC_LIBRARY,
            output_file_path=OUTPUT_DIRECTORY.joinpath("data.csv")
        )

        pipeline = BuildCSVPipeline(self._logger, self._file_handler, self._tag_extractor, self._audio_file_handler, self._audio_calculator)
        pipeline.build_pipeline()
        pipeline.flow(pipeline_context)

        self._logger.info("CSV has been successfully created.")
