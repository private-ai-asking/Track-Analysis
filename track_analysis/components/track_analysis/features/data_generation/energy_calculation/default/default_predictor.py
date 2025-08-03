import numpy as np
import pandas as pd
from scipy.special import expit

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class DefaultAudioEnergyPredictor(EnergyCalculator):
    def __init__(self, logger: HoornLogger, model: EnergyModel):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._model = model

    def calculate_energy_for_row(self, row: pd.Series) -> float:
        track_features = row[self._model.feature_names]
        if track_features.isnull().any():
            return np.nan

        features_array = track_features.values.reshape(1, -1)
        scaled_features = self._model.scaler.transform(features_array)
        pc1_score = self._model.pca.transform(scaled_features)[0, 0]

        initial_rating = self._model.spline(pc1_score)
        scaled_rating = (initial_rating - 5.5) * (10 / 9)
        sigmoid_output = expit(scaled_rating)
        final_energy = 1 + sigmoid_output * 9

        return round(float(np.clip(final_energy, 1.0, 10.0)), 1)

    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        if df_to_process.empty:
            return df_to_process

        self._logger.info(f"Calculating energy for a DataFrame with {len(df_to_process)} tracks...", separator=self._separator)
        df_copy = df_to_process.copy()
        energy_ratings = df_copy.apply(self.calculate_energy_for_row, axis=1)
        df_copy[target_column.value] = energy_ratings
        self._logger.info("Calculation complete.", separator=self._separator)

        return df_copy
