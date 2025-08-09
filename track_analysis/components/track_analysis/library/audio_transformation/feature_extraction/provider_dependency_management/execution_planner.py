from typing import List, Set, Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import (
    AudioDataFeatureProvider,
)


class ExecutionPlanner:
    """Performs a topological sort on a subset of providers to create an execution plan."""

    def __init__(self, all_providers: List[AudioDataFeatureProvider], logger: HoornLogger):
        self._feature_map = self._build_feature_to_provider_map(all_providers)
        self._logger = logger
        self._separator = self.__class__.__name__

    def generate_plan(
            self, providers_to_run: List[AudioDataFeatureProvider]
    ) -> List[AudioDataFeatureProvider]:
        """Creates an ordered execution plan for a specific set of providers."""
        order: List[AudioDataFeatureProvider] = []
        resolved: Set[AudioDataFeature] = self.get_required_base_features(
            providers_to_run
        )
        pending = providers_to_run.copy()

        stage_num = 1
        self._logger.info("--- Feature Execution Plan Analysis ---", separator=self._separator)

        while pending:
            ready_to_process = self._find_ready_providers(pending, resolved)
            if ready_to_process:
                provider_names = [p.__class__.__name__ for p in ready_to_process]
                self._logger.info(
                    f"Stage {stage_num}: Can run {len(ready_to_process)} providers in parallel -> {provider_names}",
                    separator=self._separator
                )
                stage_num += 1
            else:
                self._handle_unresolvable_graph(pending)
            self._update_state(ready_to_process, order, pending, resolved)

        return order

    def _build_feature_to_provider_map(
            self, all_providers: List[AudioDataFeatureProvider]
    ) -> Dict[AudioDataFeature, AudioDataFeatureProvider]:
        """Creates a map from each output feature to the provider that produces it."""
        feature_map = {}
        for calc in all_providers:
            for out_feature in self._get_provider_outputs(calc):
                if out_feature in feature_map:
                    raise ValueError(f"Feature '{out_feature.name}' is produced by multiple providers.")
                feature_map[out_feature] = calc
        return feature_map

    def get_required_base_features(
            self, providers: List[AudioDataFeatureProvider]
    ) -> Set[AudioDataFeature]:
        """Identifies features that must be provided as initial input for a given plan."""
        required = set()
        for calc in providers:
            for dep in calc.dependencies:
                if dep not in self._feature_map:
                    required.add(dep)
        return required

    @staticmethod
    def _are_dependencies_met(
            provider: AudioDataFeatureProvider, resolved: Set[AudioDataFeature]
    ) -> bool:
        """Checks if all dependencies for a given provider have been resolved."""
        return all(dep in resolved for dep in provider.dependencies)

    def _find_ready_providers(
            self,
            pending_providers: List[AudioDataFeatureProvider],
            resolved_features: Set[AudioDataFeature],
    ) -> List[AudioDataFeatureProvider]:
        """Finds providers from the pending list whose dependencies are met."""
        return [
            calc
            for calc in pending_providers
            if self._are_dependencies_met(calc, resolved_features)
        ]

    def _update_state(
            self,
            ready_providers: List[AudioDataFeatureProvider],
            order: List[AudioDataFeatureProvider],
            pending: List[AudioDataFeatureProvider],
            resolved: Set[AudioDataFeature],
    ) -> None:
        """Updates the state for the next iteration of the topological sort."""
        for calc in ready_providers:
            order.append(calc)
            pending.remove(calc)
            for output_feature in self._get_provider_outputs(calc):
                resolved.add(output_feature)

    def _handle_unresolvable_graph(
            self, unresolved_providers: List[AudioDataFeatureProvider]
    ) -> None:
        """Raises a comprehensive error if a cycle is detected."""
        unresolved_features = {
            f.name
            for c in unresolved_providers
            for f in self._get_provider_outputs(c)
        }
        raise RuntimeError(
            "A circular dependency was detected or a dependency is missing. "
            f"Could not resolve features: {unresolved_features}"
        )

    @staticmethod
    def _get_provider_outputs(
            calc: AudioDataFeatureProvider,
    ) -> List[AudioDataFeature]:
        """Helper to consistently handle single or multiple output features."""
        return (
            calc.output_features
            if isinstance(calc.output_features, list)
            else [calc.output_features]
        )
