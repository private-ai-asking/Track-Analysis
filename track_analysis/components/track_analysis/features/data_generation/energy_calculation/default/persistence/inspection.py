import json
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
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
        feature_names = [f.value for f in config.feature_columns]
        return {
            "metadata": {
                "model_name": config.name,
                "cache_created_utc": datetime.now(UTC).isoformat(),
                "data_hash": model.data_hash
            },
            "training": {
                "training_set_shape": model.features_shape.to_dict(),
                "feature_names": feature_names,
            },
            "pca": {
                "pca_loadings": dict(zip(feature_names, model.pca.components_[0])),
                "explained_variance_ratio": float(model.pca.explained_variance_ratio_[0]),
            },
            "quantile_anchors_pc1": {
                "low": model.spline.x[0],
                "medium": model.spline.x[1],
                "high": model.spline.x[2]
            }
        }
