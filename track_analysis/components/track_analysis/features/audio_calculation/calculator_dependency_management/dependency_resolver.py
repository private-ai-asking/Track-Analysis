from typing import List, Dict, Set

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import (
    AudioDataFeatureCalculator,
)
from track_analysis.components.track_analysis.features.audio_calculation.calculator_dependency_management.execution_planner import \
    ExecutionPlanner


class DependencyResolver:
    """Resolves the full set of required calculators and their execution order."""

    def __init__(self, all_calculators: List[AudioDataFeatureCalculator]):
        self._all_calculators = all_calculators
        self._feature_map = self._build_feature_to_calculator_map()
        self._planner = ExecutionPlanner(all_calculators)

    def get_execution_plan(
            self, features_to_calculate: List[AudioDataFeature]
    ) -> List[AudioDataFeatureCalculator]:
        """
        Determines the full set of required calculators and returns them in a valid
        topological order.
        """
        required_calculators = self._get_required_calculators(features_to_calculate)
        return self._planner.generate_plan(required_calculators)

    def get_required_base_features(
            self, calculators: List[AudioDataFeatureCalculator]
    ) -> Set[AudioDataFeature]:
        """Public method to expose the base feature requirement for a given plan."""
        return self._planner.get_required_base_features(calculators)

    # --- Private Helper Methods ---

    def _build_feature_to_calculator_map(self) -> Dict[AudioDataFeature, AudioDataFeatureCalculator]:
        feature_map = {}
        for calc in self._all_calculators:
            for out_feature in self._get_calculator_outputs(calc):
                if out_feature in feature_map:
                    raise ValueError(f"Feature '{out_feature.name}' is produced by multiple calculators.")
                feature_map[out_feature] = calc
        return feature_map

    def _get_required_calculators(
            self, features_to_calculate: List[AudioDataFeature]
    ) -> List[AudioDataFeatureCalculator]:
        all_required_features = self._resolve_all_dependencies(features_to_calculate)
        return list({
            self._feature_map[feature]
            for feature in all_required_features
            if feature in self._feature_map
        })

    def _resolve_all_dependencies(
            self, features_to_calculate: List[AudioDataFeature]
    ) -> Set[AudioDataFeature]:
        all_deps = set()
        for feature in set(features_to_calculate):
            self._recursive_dependency_find(feature, all_deps)
        return all_deps

    def _recursive_dependency_find(
            self, feature: AudioDataFeature, resolved_deps: Set[AudioDataFeature]
    ) -> None:
        if feature in resolved_deps:
            return
        resolved_deps.add(feature)
        calculator = self._feature_map.get(feature)
        if calculator:
            for dependency in calculator.dependencies:
                self._recursive_dependency_find(dependency, resolved_deps)

    @staticmethod
    def _get_calculator_outputs(
            calc: AudioDataFeatureCalculator,
    ) -> List[AudioDataFeature]:
        return (
            calc.output_features
            if isinstance(calc.output_features, list)
            else [calc.output_features]
        )
