import dataclasses
import json
import os
from pathlib import Path

import joblib
from scipy.interpolate import CubicSpline

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel, TrainingShape
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.persistence.inspection import \
    DefaultInspectionDataPersistence


@dataclasses.dataclass(frozen=True)
class ModelPaths:
    scaler_path: Path
    pca_path: Path
    spline_path: Path
    inspection_path: Path


class DefaultModelPersistence:
    """Handles persistence of binary model artifacts (scaler, PCA) and spline parameters."""

    def __init__(self, logger: HoornLogger, root_cache_dir: Path, inspection_persistence: DefaultInspectionDataPersistence):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._root_cache_dir = root_cache_dir / "energy_model"
        self._inspection_persistence = inspection_persistence

    def _get_model_cache_dir(self, config: EnergyModelConfig) -> Path:
        return self._root_cache_dir / f"{config.name}-{config.version}"

    def _get_model_paths(self, config: EnergyModelConfig, make_dirs: bool = True) -> ModelPaths:
        model_root = self._get_model_cache_dir(config)

        if make_dirs:
            os.makedirs(model_root, exist_ok=True)

        return ModelPaths(
            scaler_path=model_root / "energy_scaler.joblib",
            pca_path=model_root / "energy_pca.joblib",
            spline_path=model_root / "energy_spline_params.json",
            inspection_path=model_root / "energy_model_inspection.json",
        )

    def load(self, config: EnergyModelConfig) -> EnergyModel | None:
        model_paths: ModelPaths = self._get_model_paths(config, make_dirs=False)

        if not all(p.exists() for p in [model_paths.scaler_path, model_paths.pca_path, model_paths.spline_path, model_paths.inspection_path]):
            return None

        inspection_data = self._inspection_persistence.load(model_paths.inspection_path)
        if not inspection_data:
            return None

        try:
            scaler = joblib.load(model_paths.scaler_path)
            pca = joblib.load(model_paths.pca_path)
            with open(model_paths.spline_path, 'r') as f:
                params = json.load(f)
            spline = CubicSpline(params['x_points'], params['y_points'])

            return EnergyModel(
                scaler=scaler, pca=pca, spline=spline,
                feature_names=config.get_feature_names(),
                spline_y_points=params['y_points'],
                data_hash=inspection_data["metadata"]["data_hash"],
                features_shape=TrainingShape.from_dict(inspection_data["training"]["training_set_shape"])
            )
        except Exception as e:
            self._logger.error(f"Failed to load model artifacts for '{config.name}' @ '{config.version}': {e}", separator=self._separator)
            return None

    def save(self, model: EnergyModel, config: EnergyModelConfig) -> None:
        model_paths: ModelPaths = self._get_model_paths(config)

        try:
            joblib.dump(model.scaler, model_paths.scaler_path)
            joblib.dump(model.pca, model_paths.pca_path)
            spline_params = {'x_points': model.spline.x.tolist(), 'y_points': model.spline_y_points}
            with open(model_paths.spline_path, 'w') as f:
                json.dump(spline_params, f, indent=4)

            self._inspection_persistence.save(model_paths.inspection_path, model, config)

            self._logger.info(f"Energy model '{config.name}' @ 'v{config.version}' saved to cache.", separator=self._separator)

        except Exception as e:
            self._logger.error(f"Failed to save model '{config.name}' @ 'v{config.version}' to cache: {e}", separator=self._separator)
