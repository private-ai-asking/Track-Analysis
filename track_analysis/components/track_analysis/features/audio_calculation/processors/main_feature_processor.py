import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from enum import Enum
from pathlib import Path
from threading import Lock, Semaphore
from typing import List, Dict, Any, Tuple, Set, Iterator

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider_orchestrator import \
    AudioDataFeatureProviderOrchestrator
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class MainFeatureProcessor:
    def __init__(
            self,
            orchestrator: AudioDataFeatureProviderOrchestrator,
            logger: HoornLogger,
            max_io_workers: int = 50,
            cpu_workers: int = os.cpu_count(),
            min_workers: int = 2,
            adjustment_interval: int = 100,
    ):
        # Core components
        self._orchestrator = orchestrator
        self._logger = logger
        self._separator = "BuildCSV.MainFeatureProcessor"
        self._time_utils: TimeUtils = TimeUtils()

        # Worker parameters
        self._min_workers = min_workers
        self._max_io_workers = max_io_workers
        self._cpu_workers = cpu_workers
        self._current_num_workers = cpu_workers

        # Tuning parameters
        self._adjustment_interval = adjustment_interval
        self._io_bound_threshold = 0.25
        self._cpu_bound_threshold = 0.05

        # Concurrency and state management
        self._memory_limiter = Semaphore(self._max_io_workers)
        self._lock = Lock()

        # State for tuning intervals
        self._processed_since_adjustment: int = 0
        self._timings_since_adjustment: List[Dict[str, float]] = []

        # State for overall progress logging
        self._total_processed: int = 0
        self._total_to_process: int = 0

    def _calculate_ideal_workers(self) -> int:
        """The brain of the auto-tuner. Calculates the new optimal worker count."""
        with self._lock:
            if not self._timings_since_adjustment:
                return self._current_num_workers
            wait_times = [t['wait_time'] for t in self._timings_since_adjustment]
            process_times = [t['process_time'] for t in self._timings_since_adjustment]
            avg_wait = np.mean(wait_times)
            avg_process = np.mean(process_times)
            self._timings_since_adjustment = []

        if avg_process < 1e-6:
            wait_to_process_ratio = float('inf')
        else:
            wait_to_process_ratio = avg_wait / avg_process

        self._logger.debug(
            f"Tuning check: Wait/Process Ratio={wait_to_process_ratio:.2f} "
            f"(avg_wait={avg_wait*1000:.1f}ms, avg_process={avg_process*1000:.1f}ms)",
            separator=self._separator
        )

        if wait_to_process_ratio > self._io_bound_threshold:
            new_workers = self._current_num_workers + int((self._max_io_workers - self._current_num_workers) / 2)
            self._logger.info("Workload is I/O-bound, increasing workers.", separator=self._separator)
        elif wait_to_process_ratio < self._cpu_bound_threshold:
            new_workers = self._current_num_workers - int((self._current_num_workers - self._cpu_workers) / 2)
            self._logger.info("Workload is CPU-bound, decreasing workers.", separator=self._separator)
        else:
            return self._current_num_workers

        return max(self._min_workers, min(self._max_io_workers, new_workers))

    def _process_track(self, row_info_tuple: Tuple, requested_features: List[AudioDataFeature]) -> Dict[str, Any]:
        """
        Helper function that processes a single track, adding the audio path to the
        results for later merging.
        """
        time_before_acquire = time.perf_counter()
        self._memory_limiter.acquire()
        time_after_acquire = time.perf_counter()
        try:
            idx, row = row_info_tuple
            initial_data = {AudioDataFeature.AUDIO_PATH: Path(row[Header.Audio_Path.value])}

            calculated_features = self._orchestrator.process_track(idx, initial_data, requested_features)

            time_after_process = time.perf_counter()
            wait_time = time_after_acquire - time_before_acquire
            process_time = time_after_process - time_after_acquire

            audio_unique_identifier_col = Header.UUID.value
            calculated_features[audio_unique_identifier_col] = row[audio_unique_identifier_col]

            return {
                "result": {k.name if isinstance(k, Enum) else k: v for k, v in calculated_features.items()},
                "timing": {"wait_time": wait_time, "process_time": process_time}
            }
        finally:
            self._memory_limiter.release()

    def _handle_completed_future(self, future: Future, all_track_features: List, all_timings: List) -> None:
        """Processes the result of a completed future, handling exceptions and storing results."""
        try:
            response = future.result()
            if response and "result" in response:
                all_track_features.append(response["result"])
                all_timings.append(response["timing"])

                # Update counters and log overall progress under a single lock
                with self._lock:
                    self._timings_since_adjustment.append(response["timing"])
                    self._processed_since_adjustment += 1
                    self._total_processed += 1

                    # Log the overall progress for every completed track
                    self._logger.info(
                        f"Processed {self._total_processed} / {self._total_to_process} "
                        f"({self._total_processed / self._total_to_process * 100:.2f}%) tracks.",
                        separator=self._separator
                    )
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(f"A track failed to process: {e}\n{tb}", separator=self._separator)

    def _try_submit_new_task(self, executor: ThreadPoolExecutor, active_futures: Set[Future], tasks: Iterator, requested_features: List) -> None:
        """Tries to fetch and submit one new task from the iterator."""
        try:
            new_task_data = next(tasks)
            active_futures.add(executor.submit(self._process_track, new_task_data, requested_features))
        except StopIteration:
            pass

    def _resize_pool_if_needed(self, executor: ThreadPoolExecutor, tasks: Iterator, requested_features: List) -> Tuple[ThreadPoolExecutor, Set[Future]]:
        """Checks if the pool needs resizing and manages the recreation process."""
        if self._processed_since_adjustment < self._adjustment_interval:
            return executor, set()

        new_worker_count = self._calculate_ideal_workers()
        self._processed_since_adjustment = 0

        if new_worker_count == self._current_num_workers:
            return executor, set()

        self._logger.info(f"ADJUSTING POOL SIZE: from {self._current_num_workers} to {new_worker_count} workers.", separator=self._separator)
        executor.shutdown(wait=True)
        new_executor = ThreadPoolExecutor(max_workers=new_worker_count)
        self._current_num_workers = new_worker_count
        new_active_futures = set()
        for _ in range(self._current_num_workers):
            self._try_submit_new_task(new_executor, new_active_futures, tasks, requested_features)
        return new_executor, new_active_futures

    def _log_summary_report(self, all_timings: List[Dict[str, float]], total_duration: float, total_tracks: int) -> None:
        """Logs the final timing analysis report."""
        self._logger.info(f"--- DYNAMIC BATCH COMPLETE ---", separator=self._separator)
        self._logger.info(f"Total Tracks Processed: {len(all_timings)} / {total_tracks}", separator=self._separator)
        self._logger.info(f"Total Time Elapsed: {self._time_utils.format_time(total_duration)}", separator=self._separator)
        if total_duration > 0:
            avg_throughput = len(all_timings) / total_duration
            self._logger.info(f"Overall Throughput: {avg_throughput:.2f} tracks/second", separator=self._separator)
        if all_timings:
            timings_df = pd.DataFrame(all_timings)
            self._logger.info("--- Timing Analysis (seconds) ---", separator=self._separator)
            self._logger.info(f"\n{timings_df.describe().to_string()}", separator=self._separator)

    def process_batch(self, to_process_df: pd.DataFrame, requested_features: List[AudioDataFeature]) -> pd.DataFrame:
        """Runs the dynamic, self-tuning batch processing workload."""
        # Initialize/reset overall progress counters
        self._total_processed = 0
        self._total_to_process = len(to_process_df)

        self._logger.info(f"Starting dynamic batch of {self._total_to_process} tracks. CPU Workers={self._cpu_workers}, Max I/O Workers={self._max_io_workers}", separator=self._separator)

        all_track_features, all_timings = [], []
        tasks_iterator: Iterator = iter(to_process_df.iterrows())
        start_time = time.perf_counter()

        self._current_num_workers = self._cpu_workers
        executor = ThreadPoolExecutor(max_workers=self._current_num_workers)
        self._logger.info(f"Starting with {self._current_num_workers} workers.", separator=self._separator)

        try:
            active_futures = set()
            for _ in range(self._current_num_workers):
                self._try_submit_new_task(executor, active_futures, tasks_iterator, requested_features)

            while active_futures:
                completed_futures = as_completed(active_futures)
                for future in list(completed_futures):
                    active_futures.remove(future)
                    self._handle_completed_future(future, all_track_features, all_timings)
                    self._try_submit_new_task(executor, active_futures, tasks_iterator, requested_features)

                new_executor, new_futures = self._resize_pool_if_needed(executor, tasks_iterator, requested_features)
                if new_futures:
                    executor = new_executor
                    active_futures = new_futures
        finally:
            if executor:
                executor.shutdown(wait=True)

        total_duration = time.perf_counter() - start_time
        self._log_summary_report(all_timings, total_duration, self._total_to_process)

        return pd.DataFrame(all_track_features)
