import time
from typing import List, Dict, Any

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import (
    AudioDataFeatureProvider, )
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.execution.plan_executor import \
    PlanExecutor
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.model.track_processing_result import \
    TrackProcessingResult
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.provider_dependency_management.dependency_resolver import \
    DependencyResolver
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.validation.feature_data_validator import \
    FeatureDataValidator
from track_analysis.components.track_analysis.library.timing.timing_analysis import TimingAnalyzer


class AudioDataFeatureProviderOrchestrator:
    def __init__(self, providers: List[AudioDataFeatureProvider], logger: HoornLogger, timing_analyzer: TimingAnalyzer, feature_threads_number: int = 24):
        self._resolver = DependencyResolver(providers, logger)
        self._logger = logger
        self._separator = self.__class__.__name__
        self._time_utils: TimeUtils = TimeUtils()
        self._plan_executor: PlanExecutor = PlanExecutor(feature_threads_num=feature_threads_number, timing_analyzer=timing_analyzer, logger=logger)

    def process_track(
            self,
            track_idx: int,
            initial_data: Dict[AudioDataFeature, Any],
            features_to_calculate: List[AudioDataFeature],
    ) -> TrackProcessingResult:
        """
        Processes a single track, calculating only the requested metrics and their dependencies.
        """
        start_time = time.perf_counter()
        execution_plan, required_base_features = self._resolver.resolve(features_to_calculate)

        FeatureDataValidator.validate_initial_data(initial_data, required_base_features)
        all_results = self._plan_executor.execute_plan(initial_data, execution_plan)

        for feature in features_to_calculate:
            if feature not in all_results.retrieved_features:
                self._logger.warning(f"Requested feature '{feature}' was not found in the final results.", separator=self._separator)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        self._logger.debug(f"Finished processing track {track_idx} in: {self._time_utils.format_time(elapsed, round_digits=2)}.", separator=self._separator)

        return TrackProcessingResult(retrieved_features=all_results.retrieved_features, provider_stats=all_results.provider_stats)
