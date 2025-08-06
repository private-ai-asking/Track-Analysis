import time
from enum import Enum
from typing import List, Dict, Any, Union, Set

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import (
    AudioDataFeatureProvider,
)
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.provider_dependency_management.dependency_resolver import \
    DependencyResolver


class AudioDataFeatureProviderOrchestrator:
    def __init__(self, providers: List[AudioDataFeatureProvider], logger: HoornLogger):
        self._resolver = DependencyResolver(providers)
        self._logger = logger
        self._separator = self.__class__.__name__
        self._time_utils: TimeUtils = TimeUtils()

    # --- Execution Methods ---

    def _validate_initial_data(
            self,
            initial_data: Dict[AudioDataFeature, Any],
            execution_plan: List[AudioDataFeatureProvider],
    ) -> None:
        """Ensures all necessary base features for a specific plan are present."""
        required_base_features = self._resolver.get_required_base_features(execution_plan)
        missing_deps = required_base_features - initial_data.keys()
        if missing_deps:
            raise ValueError(
                f"Missing required initial features for this calculation: "
                f"{[dep.name for dep in missing_deps]}"
            )

    def _execute_plan(
            self,
            initial_data: Dict[AudioDataFeature, Any],
            execution_plan: List[AudioDataFeatureProvider],
    ) -> Dict[AudioDataFeature, Any]:
        """Iterates through a specific execution plan and runs each calculator."""
        results = initial_data.copy()
        for calculator in execution_plan:
            calculator_name: str = calculator.__class__.__name__
            self._validate_dependencies(calculator.dependencies, results, calculator_name)

            input_data = {dep: results[dep] for dep in calculator.dependencies}
            calculated_data = calculator.provide(input_data)

            self._validate_output(calculator.output_features, calculated_data, calculator_name)
            self._update_results(results, calculator.output_features, calculated_data)
        return results

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
            results: Dict,
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
    ) -> Dict[AudioDataFeature, Any]:
        """
        Processes a single track, calculating only the requested metrics and their dependencies.
        """
        start_time = time.time()

        self._logger.debug(f"Processing track: {track_idx}", separator=self._separator)

        self._logger.trace(f"Processing track [{track_idx}]... getting execution plan.", separator=self._separator)
        execution_plan = self._resolver.get_execution_plan(features_to_calculate)

        self._logger.trace(f"Processing track [{track_idx}]... validating initial data.", separator=self._separator)
        self._validate_initial_data(initial_data, execution_plan)

        self._logger.trace(f"Processing track [{track_idx}]... executing plan.", separator=self._separator)
        all_results = self._execute_plan(initial_data, execution_plan)

        # Filter the final dictionary to return only what the user asked for, plus initial data.
        final_results = initial_data.copy()
        for feature in features_to_calculate:
            if feature in all_results:
                final_results[feature] = all_results[feature]

        end_time = time.time()
        elapsed = end_time - start_time

        self._logger.debug(f"Finished processing track {track_idx} in: {self._time_utils.format_time(elapsed, round_digits=2)}.", separator=self._separator)

        return final_results
