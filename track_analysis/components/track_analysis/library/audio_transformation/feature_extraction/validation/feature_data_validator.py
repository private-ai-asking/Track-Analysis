from enum import Enum
from typing import Dict, Any, Set, List

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature


class FeatureDataValidator:
    """Handles the validation of data pertaining to feature provider processing."""
    @staticmethod
    def validate_initial_data(
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

    @staticmethod
    def validate_dependencies(dependencies: List[AudioDataFeature], results: Dict[AudioDataFeature, Any], calc_name: str) -> None:
        for dep in dependencies:
            if dep not in results:
                raise KeyError(
                    f"Dependency resolution failed: Calculator '{calc_name}' "
                    f"requires dependency '{dep.name}', which was not found in the results. "
                    f"Ensure the producer of '{dep.name}' ran first."
                )

    @staticmethod
    def validate_output(calculator_outputs: List[AudioDataFeature] | AudioDataFeature, results: Dict[AudioDataFeature, Any], calc_name: str) -> None:
        outputs_list = [calculator_outputs] if not isinstance(calculator_outputs, list) else calculator_outputs
        allowed_outputs_set: Set[AudioDataFeature] = set(outputs_list)

        for k in results.keys():
            if k not in allowed_outputs_set:
                raise RuntimeError(
                    f"Calculator output failed: Calculator '{calc_name}' "
                    f"gives output '{k.name if isinstance(k, Enum) else k}' which wasn't in its possible outputs."
                )
