import dataclasses
import time
from enum import Enum
from typing import List, Dict, Any, Union, Set

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import (
    AudioDataFeatureProvider, ProviderProcessingStatistics,
)
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.provider_dependency_management.dependency_resolver import \
    DependencyResolver

@dataclasses.dataclass(frozen=True)
class TrackProcessingResult:
    retrieved_features: Dict[AudioDataFeature, Any]
    provider_stats: Dict[AudioDataFeatureProvider, ProviderProcessingStatistics]


class AudioDataFeatureProviderOrchestrator:
    def __init__(self, providers: List[AudioDataFeatureProvider], logger: HoornLogger):
        self._resolver = DependencyResolver(providers, logger)
        self._logger = logger
        self._separator = self.__class__.__name__
        self._time_utils: TimeUtils = TimeUtils()

    # --- Execution Methods ---

    @staticmethod
    def _validate_initial_data(
            initial_data: Dict[AudioDataFeature, Any],
            required_base_features: Set[AudioDataFeature],
    ) -> None:
        """Ensures all necessary base features are present in the initial data."""
        missing_deps = required_base_features - initial_data.keys()
        if missing_deps:
            raise ValueError(
                f"Missing required initial features for this calculation: "
                f"{sorted([dep.name for dep in missing_deps])}"
            )

    def _execute_plan(
            self,
            initial_data: Dict[AudioDataFeature, Any],
            execution_plan: List[AudioDataFeatureProvider],
    ) -> TrackProcessingResult:
        """Iterates through a specific execution plan and runs each feature provider."""
        results = initial_data.copy()
        provider_stats: Dict[AudioDataFeatureProvider, ProviderProcessingStatistics] = {}

        for feature_provider in execution_plan:
            calculator_name: str = feature_provider.__class__.__name__
            self._validate_dependencies(feature_provider.dependencies, results, calculator_name)

            input_data = {dep: results[dep] for dep in feature_provider.dependencies}
            provided_data = feature_provider.provide(input_data)
            retrieved_features = provided_data.retrieved_features
            provider_stats[feature_provider] = provided_data.statistics

            self._validate_output(feature_provider.output_features, retrieved_features, calculator_name)
            self._update_results(results, feature_provider.output_features, retrieved_features)
        return TrackProcessingResult(retrieved_features=results, provider_stats=provider_stats)

    @staticmethod
    def _validate_dependencies(dependencies: List[AudioDataFeature], results: Dict[AudioDataFeature, Any], calc_name: str) -> None:
        for dep in dependencies:
            if dep not in results:
                raise KeyError(
                    f"Dependency resolution failed: Calculator '{calc_name}' "
                    f"requires dependency '{dep.name}', which was not found in the results. "
                    f"Ensure the producer of '{dep.name}' ran first."
                )

    @staticmethod
    def _validate_output(calculator_outputs: List[AudioDataFeature] | AudioDataFeature, results: Dict[AudioDataFeature, Any], calc_name: str) -> None:
        outputs_list = [calculator_outputs] if not isinstance(calculator_outputs, list) else calculator_outputs
        allowed_outputs_set: Set[AudioDataFeature] = set(outputs_list)

        for k in results.keys():
            if k not in allowed_outputs_set:
                raise RuntimeError(
                    f"Calculator output failed: Calculator '{calc_name}' "
                    f"gives output '{k.name if isinstance(k, Enum) else k}' which wasn't in its possible outputs."
                )

    @staticmethod
    def _update_results(
            results: Dict[AudioDataFeature, Any],
            output_features: Union[AudioDataFeature, List[AudioDataFeature]],
            calculated_data: Any,
    ) -> None:
        """Updates the results dictionary for single or multi-output calculators."""
        if isinstance(calculated_data, dict):
            results.update(calculated_data)
        else:
            results[output_features] = calculated_data

    # --- Public Method ---

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

        self._validate_initial_data(initial_data, required_base_features)

        all_results = self._execute_plan(initial_data, execution_plan)
        retrieved_features = all_results.retrieved_features

        final_results = initial_data.copy()
        for feature in features_to_calculate:
            if feature in retrieved_features:
                final_results[feature] = retrieved_features[feature]
            else:
                self._logger.warning(f"Requested feature '{feature}' was not found in the results.", separator=self._separator)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        self._logger.debug(f"Finished processing track {track_idx} in: {self._time_utils.format_time(elapsed, round_digits=2)}.", separator=self._separator)

        return TrackProcessingResult(retrieved_features=final_results, provider_stats=all_results.provider_stats)
