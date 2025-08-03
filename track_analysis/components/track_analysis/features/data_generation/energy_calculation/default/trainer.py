from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.utils import \
    get_dataframe_hash
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class DefaultEnergyModelTrainer:
    def __init__(self, logger: HoornLogger, persistence: DefaultModelPersistence):
        self._logger = logger
        self._persistence = persistence
        self._separator = self.__class__.__name__

    def train_or_load(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> Tuple[EnergyModel, bool]:
        feature_names = [f.value for f in config.feature_columns]
        features_df = training_data[feature_names].dropna()
        if features_df.empty:
            raise ValueError("No valid data for training after dropping NaNs.")

        data_hash = get_dataframe_hash(features_df)

        # 1. Delegate loading to the persistence layer
        model = self._persistence.load(config, data_hash)
        if model:
            self._logger.info(f"Loaded energy model '{config.name}' from cache.", separator=self._separator)
            return model, True

        # 2. If no model, orchestrate training
        self._logger.info(f"Cache for '{config.name}' invalid. Training new model...", separator=self._separator)
        model = self._train_pipeline(config, features_df, data_hash)

        self._logger.info("Energy model training complete.", separator=self._separator)

        return model, False

    @staticmethod
    def _train_pipeline(config: EnergyModelConfig, features_df: pd.DataFrame, data_hash: str) -> EnergyModel:
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features_df)

        pca = PCA(n_components=1)
        pca.fit(scaled_features)
        feature_names = [f.value for f in config.feature_columns]
        loudness_idx = feature_names.index(Header.Mean_RMS.value)
        if pca.components_[0, loudness_idx] < 0:
            pca.components_ = -pca.components_
        pc1_scores = pca.transform(scaled_features)

        low_q, med_q, high_q = 0.15, 0.50, 0.85
        mu_low, mu_med, mu_high = np.quantile(pc1_scores, [low_q, med_q, high_q])
        spline_x_points = [mu_low, mu_med, mu_high]
        spline_y_points = [2.5, 5.5, 8.5]
        spline = CubicSpline(spline_x_points, spline_y_points)

        return EnergyModel(
            scaler=scaler, pca=pca, spline=spline,
            feature_names=feature_names,
            spline_y_points=spline_y_points,
            data_hash=data_hash,
            features_shape=features_df.shape,
        )
