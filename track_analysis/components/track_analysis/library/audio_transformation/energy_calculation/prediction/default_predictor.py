import numpy as np
import pandas as pd
from scipy.special import expit

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel


class DefaultAudioEnergyPredictor:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._model: EnergyModel | None = None

    def set_model(self, model: EnergyModel):
        self._model = model

    def calculate_energy_for_row(self, row: pd.Series) -> float:
        if self._model is None:
            raise Exception("Training model not yet set. Make sure to call set_model before calculating energy.")

        track_features = row[self._model.feature_names]
        if track_features.isnull().any():
            return np.nan

        features_df = track_features.to_frame().T

        scaled_features = self._model.scaler.transform(features_df)

        pc1_score = self._model.pca.transform(scaled_features)[0, 0]
        initial_rating = self._model.spline(pc1_score)
        scaled_rating = (initial_rating - 5.5) * (10 / 9)
        sigmoid_output = expit(scaled_rating)
        final_energy = 1 + sigmoid_output * 9

        return round(float(np.clip(final_energy, 1.0, 10.0)), 1)

    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        """
        Calculates energy ratings and returns a DataFrame containing the
        UUID and the calculated energy.
        """
        if self._model is None:
            raise Exception("Training model not yet set. Make sure to call set_model before calculating energy.")

        if df_to_process.empty:
            return pd.DataFrame(columns=[Header.UUID.value, target_column.value])

        if Header.UUID.value not in df_to_process.columns:
            raise KeyError(f"Input DataFrame to 'calculate_ratings_for_df' must contain '{Header.UUID.value}' column.")

        df_copy = df_to_process.copy()

        features_df = df_copy[self._model.feature_names]

        scaled_features = self._model.scaler.transform(features_df)
        pc1_scores = self._model.pca.transform(scaled_features)

        initial_ratings = self._model.spline(pc1_scores.flatten())
        scaled_ratings = (initial_ratings - 5.5) * (10 / 9)
        sigmoid_outputs = expit(scaled_ratings)
        final_energies = 1 + sigmoid_outputs * 9

        final_energies = np.clip(final_energies, 1.0, 10.0).round(1)

        result_df = pd.DataFrame({
            Header.UUID.value: df_copy[Header.UUID.value],
            target_column.value: final_energies
        })

        return result_df
