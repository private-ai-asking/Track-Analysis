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

from track_analysis.components.md_common_python.py_common.logging import HoornLogger, LogType
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.features.data_generation.helpers.audio_sample_loader import \
    AudioSampleLoader
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    ProviderProcessingStatistics, AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider_orchestrator import \
    AudioDataFeatureProviderOrchestrator
from track_analysis.components.track_analysis.library.timing.model.feature_timing_data import FeatureTimingData
from track_analysis.components.track_analysis.library.timing.model.timing_data import TimingData
from track_analysis.components.track_analysis.library.timing.timing_analysis import TimingAnalyzer


@dataclasses.dataclass(frozen=True)
class TrackProcessingTimeInfo:
    total_time_spent: float
    own_waiting_time: float
    own_processing_time: float
    provider_stats: Dict[AudioDataFeatureProvider, ProviderProcessingStatistics]

    def to_timing_data(self) -> TimingData:
        feature_dissection: List[FeatureTimingData] = [
            FeatureTimingData(
                associated_feature=feature_provider.__class__.__name__,
                time_spent_processing=provider_stats.time_spent_processing,
                time_spent_waiting=provider_stats.time_spent_waiting
            ) for feature_provider, provider_stats in self.provider_stats.items()
        ]

        return TimingData(
            total_time_spent=self.total_time_spent,
            own_waiting_time=self.own_waiting_time,
            own_processing_time=self.own_processing_time,
            feature_dissection=feature_dissection,
        )


@dataclasses.dataclass(frozen=True)
class TrackProcessingResult:
    features: Dict[str, Any]
    processing_info: TrackProcessingTimeInfo


class MainFeatureProcessor:
    def __init__(
            self,
            orchestrator: AudioDataFeatureProviderOrchestrator,
            logger: HoornLogger,
            timing_analyzer: TimingAnalyzer,
            cpu_workers: int = os.cpu_count(),
    ):
        # Core components
        self._orchestrator = orchestrator
        self._logger = logger
        self._separator = "BuildCSV.MainFeatureProcessor"
        self._time_utils: TimeUtils = TimeUtils()
        self._timing_analyzer = timing_analyzer

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

        with executor:
            active_futures = set()
            for _ in range(self._cpu_workers):
                self._try_submit_new_task(executor, active_futures, tasks_iterator, requested_features)

            while active_futures:
                completed_futures = as_completed(active_futures)
                for future in list(completed_futures):
                    active_futures.remove(future)
                    self._handle_completed_future(future, all_track_features, all_timings)
                    self._try_submit_new_task(executor, active_futures, tasks_iterator, requested_features)

        total_duration = time.perf_counter() - start_time
        self._log_summary_report(all_timings, total_duration, self._total_to_process)

        return pd.DataFrame(all_track_features)

    def _process_track(self,
                       row_info_tuple: Tuple,
                       requested_features: List[AudioDataFeature],
                       audio_sample_future: Future,
                       audio_path: Path) -> TrackProcessingResult | None:
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
            initial_data = {
                AudioDataFeature.AUDIO_PATH: audio_path,
                AudioDataFeature.AUDIO_SAMPLES_FUTURE: audio_sample_future,
            }

            process_time = time.perf_counter() - start_process
            track_processing_result = self._orchestrator.process_track(idx, initial_data, requested_features)

            start_process = time.perf_counter()
            retrieved_features = track_processing_result.retrieved_features

            retrieved_features[self._audio_uid_column] = row[self._audio_uid_column]
            features_dict = {k.name if isinstance(k, Enum) else k: v for k, v in retrieved_features.items()}
            process_time = process_time + (time.perf_counter() - start_process)
            total_time = time.perf_counter() - start_wait
            return TrackProcessingResult(
                features=features_dict,
                processing_info=TrackProcessingTimeInfo(
                    own_waiting_time=process_time,
                    own_processing_time=wait_time,
                    provider_stats=track_processing_result.provider_stats,
                    total_time_spent=total_time,
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

            _, row = new_task_data
            audio_path = Path(row[Header.Audio_Path.value])

            audio_sample_future: Future = executor.submit(AudioSampleLoader.load_audio_samples, audio_path)
            active_futures.add(executor.submit(self._process_track, new_task_data, requested_features, audio_sample_future, audio_path))
        except StopIteration:
            pass

    def _log_summary_report(self, all_timings: List[TrackProcessingTimeInfo], total_duration: float, total_tracks: int) -> None:
        self._logger.info(f"--- DYNAMIC BATCH COMPLETE ---", separator=self._separator)
        self._logger.info(f"Total Tracks Processed: {len(all_timings)} / {total_tracks}", separator=self._separator)

        if all_timings:
            self._timing_analyzer.analyze_time(
                timing_data_batch=[timing.to_timing_data() for timing in all_timings],
                wall_clock_time=total_duration,
                worker_count=self._cpu_workers,
                log_mode=LogType.DEBUG
            )
