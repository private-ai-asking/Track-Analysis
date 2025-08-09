from typing import List, Dict, Set, Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import (
    AudioDataFeatureProvider,
)
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.provider_dependency_management.execution_planner import \
    ExecutionPlanner


class DependencyResolver:
    """Resolves the full set of required providers and their execution order."""

    def __init__(self, all_providers: List[AudioDataFeatureProvider], logger: HoornLogger):
        self._all_providers = all_providers
        self._feature_map = self._build_feature_to_provider_map()
        self._planner = ExecutionPlanner(all_providers, logger)

    def resolve(
            self, features_to_calculate: List[AudioDataFeature]
    ) -> Tuple[List[AudioDataFeatureProvider], Set[AudioDataFeature]]:
        """
        Resolves the full dependency graph for a feature list.

        Returns a tuple containing:
        1. An ordered execution plan (list of providers).
        2. A complete set of required base features.
        """
        # 1. Find all features needed for the calculation (including dependencies).
        all_required_features = self._resolve_all_dependencies(features_to_calculate)

        # 2. Determine which required features are "base features" (they have no provider).
        #    This is the crucial step that was missing.
        required_base_features = {
            feature for feature in all_required_features
            if feature not in self._feature_map
        }

        # 3. Get the list of providers needed to create the derivable features.
        required_providers = list({
            self._feature_map[feature]
            for feature in all_required_features
            if feature in self._feature_map
        })

        # 4. Create the final, ordered execution plan from the providers.
        execution_plan = self._planner.generate_plan(required_providers)

        return execution_plan, required_base_features

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
