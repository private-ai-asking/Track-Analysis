import json
from datetime import datetime, UTC
from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig


class DefaultInspectionDataPersistence:
    """Handles the creation and persistence of human-readable inspection data."""

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

    def load(self, path: Path) -> dict | None:
        """Loads the inspection JSON file from a given path."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self._logger.error(f"Failed to load inspection data from {path}: {e}", separator=self._separator)
            return None

    def save(self, path: Path, model: EnergyModel, config: EnergyModelConfig) -> None:
        """Creates and saves the inspection data to a given path."""
        try:
            inspection_data = self._create_data_dict(model, config)
            with open(path, 'w') as f:
                json.dump(inspection_data, f, indent=4)
        except Exception as e:
            self._logger.error(f"Failed to save inspection data to {path}: {e}", separator=self._separator)

    @staticmethod
    def _create_data_dict(model: EnergyModel, config: EnergyModelConfig) -> dict:
        """Creates the human-readable inspection JSON dictionary."""
        feature_names = config.get_feature_names()
        num_components = model.number_of_pca_components

        # Pre-slice the relevant PCA data
        component_loadings = model.pca.components_[:num_components]
        variance_ratios = model.pca.explained_variance_ratio_[:num_components]

        # Create a list where each item is a dictionary detailing one component
        principal_components = [
            {
                "component": f"PC{i + 1}",
                "explained_variance_ratio": round(variance_ratios[i], 5),
                "loadings": {
                    name: round(loading, 5)
                    for name, loading in zip(feature_names, component_loadings[i])
                }
            }
            for i in range(num_components)
        ]

        return {
            "metadata": {
                "model_name": config.name,
                "model_version": config.version,
                "cache_created_utc": datetime.now(UTC).isoformat(),
                "data_hash": model.data_hash,
                "number_pca_components": num_components,
                "cumulative_variance": model.cumulative_variance,
            },
            "training": {
                "training_set_shape": model.features_shape.to_dict(),
                "feature_names": feature_names,
            },
            "principal_components": principal_components,
            "composite_score_quantile_anchors": {
                "low": model.spline.x[0],
                "medium": model.spline.x[1],
                "high": model.spline.x[2]
            }
        }
