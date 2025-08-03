import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.interpolate import CubicSpline

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.inspection import \
    DefaultInspectionDataPersistence


class DefaultModelPersistence:
    """Handles persistence of binary model artifacts (scaler, PCA) and spline parameters."""

    def __init__(self, logger: HoornLogger, root_cache_dir: Path, inspection_persistence: DefaultInspectionDataPersistence):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._root_cache_dir = root_cache_dir / "energy_model"
        self._inspection_persistence = inspection_persistence

    def load(self, config: EnergyModelConfig, data_hash: str) -> EnergyModel | None:
        model_cache_dir = self._root_cache_dir / config.name
        scaler_path = model_cache_dir / "energy_scaler.joblib"
        pca_path = model_cache_dir / "energy_pca.joblib"
        spline_path = model_cache_dir / "energy_spline_params.json"
        inspection_path = model_cache_dir / "energy_model_inspection.json"

        if not all(p.exists() for p in [scaler_path, pca_path, spline_path, inspection_path]):
            return None

        # Delegate loading the inspection file
        inspection_data = self._inspection_persistence.load(inspection_path)
        if not inspection_data or inspection_data["metadata"]["data_hash"] != data_hash:
            self._logger.warning(f"Data hash mismatch or invalid inspection file for model '{config.name}'.",
                                 separator=self._separator)
            return None

        try:
            scaler = joblib.load(scaler_path)
            pca = joblib.load(pca_path)
            with open(spline_path, 'r') as f:
                params = json.load(f)
            spline = CubicSpline(params['x_points'], params['y_points'])

            return EnergyModel(
                scaler=scaler, pca=pca, spline=spline,
                feature_names=[f.value for f in config.feature_columns],
                spline_y_points=params['y_points'],
                data_hash=data_hash,
                features_shape=inspection_data["training"]["training_set_shape"]
            )
        except Exception as e:
            self._logger.error(f"Failed to load model artifacts for '{config.name}': {e}", separator=self._separator)
            return None

    def save(self, model: EnergyModel, config: EnergyModelConfig) -> None:
        model_cache_dir = self._root_cache_dir / config.name
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = model_cache_dir / "energy_scaler.joblib"
        pca_path = model_cache_dir / "energy_pca.joblib"
        spline_path = model_cache_dir / "energy_spline_params.json"
        inspection_path = model_cache_dir / "energy_model_inspection.json"

        try:
            # Handle binary artifacts
            joblib.dump(model.scaler, scaler_path)
            joblib.dump(model.pca, pca_path)
            spline_params = {'x_points': model.spline.x.tolist(), 'y_points': model.spline_y_points}
            with open(spline_path, 'w') as f:
                json.dump(spline_params, f, indent=4)

            # Delegate saving the inspection file
            self._inspection_persistence.save(inspection_path, model, config)

            self._logger.info(f"Energy model '{config.name}' saved to cache.", separator=self._separator)

        except Exception as e:
            self._logger.error(f"Failed to save model '{config.name}' to cache: {e}", separator=self._separator)
