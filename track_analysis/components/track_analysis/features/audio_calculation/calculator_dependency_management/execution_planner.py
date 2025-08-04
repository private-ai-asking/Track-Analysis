from typing import List, Set, Dict

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import (
    AudioDataFeatureCalculator,
)


class ExecutionPlanner:
    """Performs a topological sort on a subset of calculators to create an execution plan."""

    def __init__(self, all_calculators: List[AudioDataFeatureCalculator]):
        self._feature_map = self._build_feature_to_calculator_map(all_calculators)

    def generate_plan(
            self, calculators_to_run: List[AudioDataFeatureCalculator]
    ) -> List[AudioDataFeatureCalculator]:
        """Creates an ordered execution plan for a specific set of calculators."""
        order: List[AudioDataFeatureCalculator] = []
        resolved: Set[AudioDataFeature] = self.get_required_base_features(
            calculators_to_run
        )
        pending = calculators_to_run.copy()

        while pending:
            ready_to_process = self._find_ready_calculators(pending, resolved)
            if not ready_to_process:
                self._handle_unresolvable_graph(pending)
            self._update_state(ready_to_process, order, pending, resolved)

        return order

    def _build_feature_to_calculator_map(
            self, all_calculators: List[AudioDataFeatureCalculator]
    ) -> Dict[AudioDataFeature, AudioDataFeatureCalculator]:
        """Creates a map from each output feature to the calculator that produces it."""
        feature_map = {}
        for calc in all_calculators:
            for out_feature in self._get_calculator_outputs(calc):
                if out_feature in feature_map:
                    raise ValueError(f"Feature '{out_feature.name}' is produced by multiple calculators.")
                feature_map[out_feature] = calc
        return feature_map

    def get_required_base_features(
            self, calculators: List[AudioDataFeatureCalculator]
    ) -> Set[AudioDataFeature]:
        """Identifies features that must be provided as initial input for a given plan."""
        required = set()
        for calc in calculators:
            for dep in calc.dependencies:
                if dep not in self._feature_map:
                    required.add(dep)
        return required

    @staticmethod
    def _are_dependencies_met(
            calculator: AudioDataFeatureCalculator, resolved: Set[AudioDataFeature]
    ) -> bool:
        """Checks if all dependencies for a given calculator have been resolved."""
        return all(dep in resolved for dep in calculator.dependencies)

    def _find_ready_calculators(
            self,
            pending_calculators: List[AudioDataFeatureCalculator],
            resolved_features: Set[AudioDataFeature],
    ) -> List[AudioDataFeatureCalculator]:
        """Finds calculators from the pending list whose dependencies are met."""
        return [
            calc
            for calc in pending_calculators
            if self._are_dependencies_met(calc, resolved_features)
        ]

    def _update_state(
            self,
            ready_calculators: List[AudioDataFeatureCalculator],
            order: List[AudioDataFeatureCalculator],
            pending: List[AudioDataFeatureCalculator],
            resolved: Set[AudioDataFeature],
    ) -> None:
        """Updates the state for the next iteration of the topological sort."""
        for calc in ready_calculators:
            order.append(calc)
            pending.remove(calc)
            for output_feature in self._get_calculator_outputs(calc):
                resolved.add(output_feature)

    def _handle_unresolvable_graph(
            self, unresolved_calculators: List[AudioDataFeatureCalculator]
    ) -> None:
        """Raises a comprehensive error if a cycle is detected."""
        unresolved_features = {
            f.name
            for c in unresolved_calculators
            for f in self._get_calculator_outputs(c)
        }
        raise RuntimeError(
            "A circular dependency was detected or a dependency is missing. "
            f"Could not resolve features: {unresolved_features}"
        )

    @staticmethod
    def _get_calculator_outputs(
            calc: AudioDataFeatureCalculator,
    ) -> List[AudioDataFeature]:
        """Helper to consistently handle single or multiple output features."""
        return (
            calc.output_features
            if isinstance(calc.output_features, list)
            else [calc.output_features]
        )
