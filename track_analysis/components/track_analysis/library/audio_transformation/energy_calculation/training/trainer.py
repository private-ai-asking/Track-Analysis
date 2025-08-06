import dataclasses

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel, TrainingShape
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.postprocessing.energy_model_analyzer import \
    EnergyModelAnalyzer
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.processing.energy_model_processor import \
    EnergyModelProcessor
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.utils import \
    get_dataframe_hash

@dataclasses.dataclass(frozen=True)
class PCAData:
    num_components: int
    cumulative_variance: float

class DefaultEnergyModelTrainer:
    def __init__(self, logger: HoornLogger, persistence: DefaultModelPersistence, energy_calculator: EnergyModelProcessor):
        self._logger = logger
        self._persistence = persistence
        self._separator = self.__class__.__name__
        self._energy_calculator = energy_calculator
        self._energy_model_analyzer = EnergyModelAnalyzer(logger)

    def train(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        features_df = training_data[config.get_feature_names()].dropna()
        if features_df.empty:
            raise ValueError("No valid data for training after dropping NaNs.")

        data_hash = get_dataframe_hash(features_df)
        model = self._train_pipeline(config, features_df, data_hash)

        self._logger.info("Energy model training complete.", separator=self._separator)

        return model

    def _train_pipeline(self, config: EnergyModelConfig, features_df: pd.DataFrame, data_hash: str) -> EnergyModel:
        scaler = RobustScaler()
        scaled_features = self._energy_calculator.scale_features(scaler, features_df, train=True)

        pca = PCA()
        pca.fit(scaled_features)
        loudness_idx = config.get_feature_names().index(Header.Mean_RMS.value)

        pca_data = self._get_pca_data(pca, config.cumulative_pca_variance_threshold)
        self._ensure_correct_sign_for_components(pca, loudness_idx, pca_data.num_components)

        raw_pca_scores = self._energy_calculator.get_pca_scores(pca, scaled_features)
        weighted_scores = self._energy_calculator.compute_composite_score(pca, raw_pca_scores, pca_data.num_components, pca_data.cumulative_variance)

        low_q, med_q, high_q = 0.15, 0.50, 0.85
        mu_low, mu_med, mu_high = np.quantile(weighted_scores, [low_q, med_q, high_q])
        spline_x_points = [mu_low, mu_med, mu_high]
        spline_y_points = [2.5, 5.5, 8.5]
        spline = CubicSpline(spline_x_points, spline_y_points)

        trained_model = EnergyModel(
            scaler=scaler, pca=pca, spline=spline,
            feature_names=config.get_feature_names(),
            spline_y_points=spline_y_points,
            data_hash=data_hash,
            features_shape=TrainingShape(features_df.shape[0], features_df.shape[1]),
            number_of_pca_components=pca_data.num_components,
            cumulative_variance=pca_data.cumulative_variance,
        )

        self._energy_model_analyzer.analyze(trained_model)

        return trained_model

    def _get_pca_data(self, pca: PCA, variance_threshold: float) -> PCAData:
        necessary: int = 0
        current_cumulative_variance: float = 0.0

        for i in range(0, pca.n_components_):
            necessary += 1
            variance = pca.explained_variance_ratio_[i]
            current_cumulative_variance += variance

            if not self._need_further_scores(variance_threshold, current_cumulative_variance):
                break

        return PCAData(necessary, current_cumulative_variance)

    @staticmethod
    def _need_further_scores(threshold_ratio: float, current_cumulative_variance: float) -> bool:
        return current_cumulative_variance <= threshold_ratio

    def _ensure_correct_sign_for_components(self, pca: PCA, loudness_idx: int, n_components: int) -> None:
        for i in range(0, n_components):
            self._ensure_correct_sign_pca_component(pca, loudness_idx, i)

    @staticmethod
    def _ensure_correct_sign_pca_component(pca: PCA, loudness_idx: int, pca_component_idx: int) -> None:
        if pca.components_[pca_component_idx, loudness_idx] < 0:
            pca.components_[pca_component_idx] = -pca.components_[pca_component_idx]

