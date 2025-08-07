import dataclasses
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from enum import Enum
from pathlib import Path
from threading import Lock, Semaphore
from typing import List, Dict, Any, Tuple, Set, Iterator

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider_orchestrator import \
    AudioDataFeatureProviderOrchestrator

@dataclasses.dataclass(frozen=True)
class TrackProcessingTimeInfo:
    overall_processing_time: float
    overall_waiting_time: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "overall_processing_time": self.overall_processing_time,
            "overall_waiting_time": self.overall_waiting_time,
        }


@dataclasses.dataclass(frozen=True)
class TrackProcessingResult:
    features: Dict[str, Any]
    processing_info: TrackProcessingTimeInfo


class MainFeatureProcessor:
    def __init__(
            self,
            orchestrator: AudioDataFeatureProviderOrchestrator,
            logger: HoornLogger,
            cpu_workers: int = os.cpu_count(),
    ):
        # Core components
        self._orchestrator = orchestrator
        self._logger = logger
        self._separator = "BuildCSV.MainFeatureProcessor"
        self._time_utils: TimeUtils = TimeUtils()

        # Worker parameters
        self._cpu_workers = cpu_workers
        self._thread_buffer_amount = self._cpu_workers * 2

        # Concurrency and state management
        self._memory_limiter = Semaphore(self._thread_buffer_amount)
        self._lock = Lock()

        # State for overall progress logging
        self._total_processed: int = 0
        self._total_to_process: int = 0

        self._audio_uid_column = Header.UUID.value

    def _process_track(self, row_info_tuple: Tuple, requested_features: List[AudioDataFeature]) -> TrackProcessingResult | None:
        """
        Helper function that processes a single track, adding the audio path to the
        results for later merging.
        """
        start_wait = time.perf_counter()
        self._memory_limiter.acquire()
        start_process = time.perf_counter()

        wait_time = start_process - start_wait

        try:
            idx, row = row_info_tuple
            initial_data = {AudioDataFeature.AUDIO_PATH: Path(row[Header.Audio_Path.value])}

            retrieved_features = self._orchestrator.process_track(idx, initial_data, requested_features)

            retrieved_features[self._audio_uid_column] = row[self._audio_uid_column]
            features_dict = {k.name if isinstance(k, Enum) else k: v for k, v in retrieved_features.items()}

            time_after_process = time.perf_counter()
            process_time = time_after_process - start_process
            return TrackProcessingResult(
                features=features_dict,
                processing_info=TrackProcessingTimeInfo(
                    overall_processing_time=process_time,
                    overall_waiting_time=wait_time
                )
            )
        finally:
            self._memory_limiter.release()

    def _handle_completed_future(self, future: Future, all_track_features: List[Dict[str, Any]], all_timings: List[TrackProcessingTimeInfo]) -> None:
        """Processes the result of a completed future, handling exceptions and storing results."""
        try:
            response: TrackProcessingResult | None = future.result()
            if response:
                all_track_features.append(response.features)
                all_timings.append(response.processing_info)

                with self._lock:
                    self._total_processed += 1

                    self._logger.info(
                        f"Processed {self._total_processed} / {self._total_to_process} "
                        f"({self._total_processed / self._total_to_process * 100:.2f}%) tracks.",
                        separator=self._separator
                    )
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(f"A track failed to process: {e}\n{tb}", separator=self._separator)

    def _try_submit_new_task(self, executor: ThreadPoolExecutor, active_futures: Set[Future], tasks: Iterator, requested_features: List[AudioDataFeature]) -> None:
        """Tries to fetch and submit one new task from the iterator."""
        try:
            new_task_data = next(tasks)
            active_futures.add(executor.submit(self._process_track, new_task_data, requested_features))
        except StopIteration:
            pass

    def _log_summary_report(self, all_timings: List[TrackProcessingTimeInfo], total_duration: float, total_tracks: int) -> None:
        """Logs the final timing analysis report."""
        self._logger.info(f"--- DYNAMIC BATCH COMPLETE ---", separator=self._separator)
        self._logger.info(f"Total Tracks Processed: {len(all_timings)} / {total_tracks}", separator=self._separator)
        self._logger.info(f"Total Time Elapsed: {self._time_utils.format_time(total_duration)}", separator=self._separator)
        if total_duration > 0:
            avg_throughput = len(all_timings) / total_duration
            self._logger.info(f"Overall Throughput: {avg_throughput:.2f} tracks/second", separator=self._separator)
        if all_timings:
            timings_df = pd.DataFrame([timing.to_dict() for timing in all_timings])
            self._logger.info("--- Timing Analysis (seconds) ---", separator=self._separator)
            self._logger.info(f"\n{timings_df.describe().to_string()}", separator=self._separator)

    def process_batch(self, to_process_df: pd.DataFrame, requested_features: List[AudioDataFeature]) -> pd.DataFrame:
        """Runs the dynamic, self-tuning batch processing workload."""
        self._total_processed = 0
        self._total_to_process = len(to_process_df)

        self._logger.info(f"Starting dynamic batch of {self._total_to_process} tracks. CPU Workers={self._cpu_workers}, Thread Buffer (For I/O)={self._thread_buffer_amount}", separator=self._separator)

        all_track_features: List[Dict[str, Any]] = []
        all_timings: List[TrackProcessingTimeInfo] = []

        tasks_iterator: Iterator = iter(to_process_df.iterrows())
        start_time = time.perf_counter()

        executor = ThreadPoolExecutor(max_workers=self._cpu_workers)

        try:
            active_futures = set()
            for _ in range(self._cpu_workers):
                self._try_submit_new_task(executor, active_futures, tasks_iterator, requested_features)

            while active_futures:
                completed_futures = as_completed(active_futures)
                for future in list(completed_futures):
                    active_futures.remove(future)
                    self._handle_completed_future(future, all_track_features, all_timings)
                    self._try_submit_new_task(executor, active_futures, tasks_iterator, requested_features)
        finally:
            if executor:
                executor.shutdown(wait=True)

        total_duration = time.perf_counter() - start_time
        self._log_summary_report(all_timings, total_duration, self._total_to_process)

        return pd.DataFrame(all_track_features)
