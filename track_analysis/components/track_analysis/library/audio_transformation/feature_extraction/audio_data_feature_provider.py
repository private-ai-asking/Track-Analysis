import dataclasses
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Dict, Any
import threading

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult


@dataclasses.dataclass(frozen=True)
class ProviderProcessingStatistics:
    time_spent_processing: float
    time_spent_waiting: float


@dataclasses.dataclass(frozen=True)
class ProviderResult:
    retrieved_features: Dict[AudioDataFeature, Any]
    statistics: ProviderProcessingStatistics


class AudioDataFeatureProvider(ABC):
    """
    A base class for a single-feature provider.
    This implementation is thread-safe using thread-local storage, requiring
    no changes to concrete subclass implementations.
    """
    def __init__(self):
        self._thread_local = threading.local()

    @property
    @abstractmethod
    def dependencies(self) -> List[AudioDataFeature]:
        """A list of AudioDataFeature enums that this provider depends on."""
        pass

    @property
    @abstractmethod
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        """The AudioDataFeature enum for the feature this provider produces."""
        pass

    @abstractmethod
    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        """Performs the core logic for providing features."""
        pass

    def __validate_state(self):
        is_processing = getattr(self._thread_local, 'is_processing', False)
        is_waiting = getattr(self._thread_local, 'is_waiting', False)
        if is_processing or is_waiting:
            raise RuntimeError("You cannot have multiple states at the same time, whether they are processing or waiting or any combination.")

    @contextmanager
    def _measure_processing(self):
        self.__validate_state()
        self._thread_local.is_processing = True
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._thread_local.timings["processing"].append(duration)
            self._thread_local.is_processing = False

    @contextmanager
    def _measure_waiting(self):
        self.__validate_state()
        self._thread_local.is_waiting = True
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._thread_local.timings["waiting"].append(duration)
            self._thread_local.is_waiting = False

    def _add_timed_cache_times(self, timed_cache: TimedCacheResult[Any]):
        """Parses a timed cache result to update thread-local timing state."""
        self.__validate_state()
        self._thread_local.timings["waiting"].append(timed_cache.time_waiting)
        self._thread_local.timings["processing"].append(timed_cache.time_processing)

    def provide(self, data: Dict[AudioDataFeature, Any]) -> ProviderResult:
        """
        Performs the retrieval operation for a single track.
        This method is now thread-safe.
        """
        self._thread_local.timings = {"processing": [], "waiting": []}
        self._thread_local.is_processing = False
        self._thread_local.is_waiting = False

        try:
            retrieved_features = self._provide(data)

            total_time_processing = np.sum(self._thread_local.timings["processing"])
            total_time_waiting = np.sum(self._thread_local.timings["waiting"])

            return ProviderResult(
                retrieved_features=retrieved_features,
                statistics=ProviderProcessingStatistics(
                    time_spent_processing=total_time_processing,
                    time_spent_waiting=total_time_waiting,
                )
            )
        finally:
            del self._thread_local.timings
            del self._thread_local.is_processing
            del self._thread_local.is_waiting
