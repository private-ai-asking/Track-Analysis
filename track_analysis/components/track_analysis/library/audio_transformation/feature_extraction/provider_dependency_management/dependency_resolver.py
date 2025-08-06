from typing import List, Dict, Set

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import (
    AudioDataFeatureProvider,
)
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.provider_dependency_management.execution_planner import \
    ExecutionPlanner


class DependencyResolver:
    """Resolves the full set of required providers and their execution order."""

    def __init__(self, all_providers: List[AudioDataFeatureProvider]):
        self._all_providers = all_providers
        self._feature_map = self._build_feature_to_provider_map()
        self._planner = ExecutionPlanner(all_providers)

    def get_execution_plan(
            self, features_to_calculate: List[AudioDataFeature]
    ) -> List[AudioDataFeatureProvider]:
        """
        Determines the full set of required providers and returns them in a valid
        topological order.
        """
        required_providers = self._get_required_providers(features_to_calculate)
        return self._planner.generate_plan(required_providers)

    def get_required_base_features(
            self, providers: List[AudioDataFeatureProvider]
    ) -> Set[AudioDataFeature]:
        """Public method to expose the base feature requirement for a given plan."""
        return self._planner.get_required_base_features(providers)

    # --- Private Helper Methods ---

    def _build_feature_to_provider_map(self) -> Dict[AudioDataFeature, AudioDataFeatureProvider]:
        """Creates a map from each output feature to the provider that produces it."""
        feature_map: Dict[AudioDataFeature, AudioDataFeatureProvider] = {}
        for calc in self._all_providers:
            for out_feature in self._get_provider_outputs(calc):
                if out_feature in feature_map:
                    existing_provider = feature_map[out_feature].__class__.__name__
                    new_provider = calc.__class__.__name__
                    raise ValueError(
                        f"Feature '{out_feature.name}' is produced by multiple providers: "
                        f"'{existing_provider}' and '{new_provider}'."
                    )
                feature_map[out_feature] = calc
        return feature_map

    def _get_required_providers(
            self, features_to_calculate: List[AudioDataFeature]
    ) -> List[AudioDataFeatureProvider]:
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
        provider = self._feature_map.get(feature)
        if provider:
            for dependency in provider.dependencies:
                self._recursive_dependency_find(dependency, resolved_deps)

    @staticmethod
    def _get_provider_outputs(
            calc: AudioDataFeatureProvider,
    ) -> List[AudioDataFeature]:
        return (
            calc.output_features
            if isinstance(calc.output_features, list)
            else [calc.output_features]
        )
