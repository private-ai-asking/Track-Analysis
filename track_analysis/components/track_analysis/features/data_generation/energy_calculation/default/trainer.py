import hashlib
import json
from datetime import datetime, UTC
from typing import List

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class DefaultEnergyModelTrainer:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._feature_columns: List[Header] = [
            # Loudness & Dynamics
            Header.Integrated_LUFS, Header.Mean_RMS, Header.Percentile_90_RMS,
            Header.Program_Dynamic_Range_LRA, Header.Crest_Factor, Header.RMS_IQR,
            # Rhythm
            Header.BPM, Header.Onset_Rate, Header.Onset_Rate_Kick,
            Header.Onset_Rate_Snare, Header.Onset_Rate_Hi_Hat,
            # Timbre & Texture
            Header.Onset_Env_Mean, Header.Onset_Env_Mean_Kick, Header.Onset_Env_Mean_Snare,
            Header.Onset_Env_Mean_Low_Mid, Header.Onset_Env_Mean_Hi_Hat,
            Header.Spectral_Centroid_Mean, Header.Spectral_Contrast_Mean, Header.Zero_Crossing_Rate_Mean,
            Header.Spectral_Flatness_Mean, Header.Spectral_Flux_Mean,
        ]
        self._feature_names = [f.value for f in self._feature_columns]

        self._cache_dir = CACHE_DIRECTORY / "energy_model"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._scaler_path = self._cache_dir / "energy_scaler.joblib"
        self._pca_path = self._cache_dir / "energy_pca.joblib"
        self._spline_params_path = self._cache_dir / "energy_spline_params.json"
        self._inspection_path = self._cache_dir / "energy_model_inspection.json"

    def train_or_load(self, training_data: pd.DataFrame) -> EnergyModel:
        """
        Main entry point to either load a valid cached model or train a new one.
        """
        features_df = training_data[self._feature_names].dropna()
        if features_df.empty:
            raise ValueError("No valid data for training after dropping NaNs.")

        data_hash = self._get_data_hash(features_df)
        model = self._load_from_cache(data_hash)

        if model:
            self._logger.info(f"Loaded energy model from cache at {self._cache_dir}", separator=self._separator)
            return model

        self._logger.info("Cache not found or invalid. Training new energy model...", separator=self._separator)
        return self._train_new_model(features_df)

    def _train_new_model(self, features_df: pd.DataFrame) -> EnergyModel:
        # 1. Scale
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features_df)

        # 2. PCA
        pca = PCA(n_components=1)
        pca.fit(scaled_features)
        loudness_idx = self._feature_names.index(Header.Mean_RMS.value)
        if pca.components_[0, loudness_idx] < 0:
            pca.components_ = -pca.components_
        pc1_scores = pca.transform(scaled_features)

        # 3. Rating Curve
        low_q, med_q, high_q = 0.15, 0.50, 0.85
        mu_low, mu_med, mu_high = np.quantile(pc1_scores, [low_q, med_q, high_q])
        spline_x_points = [mu_low, mu_med, mu_high]
        spline_y_points = [2.5, 5.5, 8.5]
        spline = CubicSpline(spline_x_points, spline_y_points)

        model = EnergyModel(scaler=scaler, pca=pca, spline=spline, feature_names=self._feature_names, spline_y_points=spline_y_points)

        # 4. Save to cache
        self._save_to_cache(model, features_df)
        self._logger.info("Energy model training complete.", separator=self._separator)

        return model

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    def _load_from_cache(self, current_data_hash: str) -> EnergyModel | None:
        if not all(p.exists() for p in [self._scaler_path, self._pca_path, self._spline_params_path, self._inspection_path]):
            return None
        try:
            with open(self._inspection_path, 'r') as f:
                inspection_data = json.load(f)
            if inspection_data["metadata"]["data_hash"] != current_data_hash:
                self._logger.warning("Data hash mismatch. Cache is stale.", separator=self._separator)
                return None

            scaler = joblib.load(self._scaler_path)
            pca = joblib.load(self._pca_path)
            with open(self._spline_params_path, 'r') as f:
                params = json.load(f)
            spline = CubicSpline(params['x_points'], params['y_points'])

            # Reconstruct the full model object, including the y_points
            return EnergyModel(
                scaler=scaler, pca=pca, spline=spline,
                feature_names=self._feature_names,
                spline_y_points=params['y_points']
            )
        except Exception as e:
            self._logger.error(f"Failed to load model from cache: {e}", separator=self._separator)
            return None

    def _save_to_cache(self, model: EnergyModel, features_df: pd.DataFrame) -> None:
        try:
            joblib.dump(model.scaler, self._scaler_path)
            joblib.dump(model.pca, self._pca_path)

            spline_params = {'x_points': model.spline.x.tolist(), 'y_points': model.spline_y_points}
            with open(self._spline_params_path, 'w') as f:
                json.dump(spline_params, f, indent=4)

            # Create inspection data
            inspection_data = self._create_inspection_data(model, features_df)
            with open(self._inspection_path, 'w') as f:
                json.dump(inspection_data, f, indent=4)
            self._logger.info(f"Energy model saved to cache.", separator=self._separator)

        except Exception as e:
            self._logger.error(f"Failed to save model to cache: {e}", separator=self._separator)

    def _create_inspection_data(self, model: EnergyModel, features_df: pd.DataFrame) -> dict:
        return {
            "metadata": {"cache_created_utc": datetime.now(UTC).isoformat(), "data_hash": self._get_data_hash(features_df)},
            "training": {"training_set_shape": list(features_df.shape), "feature_names": self._feature_names},
            "pca": {
                "pca_loadings": dict(zip(self._feature_names, model.pca.components_[0])),
                "explained_variance_ratio": float(model.pca.explained_variance_ratio_[0]),
            },
            "quantile_anchors_pc1": {"low": model.spline.x[0], "medium": model.spline.x[1], "high": model.spline.x[2]}
        }
