import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.prediction.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.preprocessing.energy_preparer import \
    EnergyDataPreparer
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class DataFramePredictionPipeline:
    """
    Handles the preparation, validation, and prediction for a DataFrame of data.
    """

    def __init__(self,
                 logger: HoornLogger,
                 data_preparer: EnergyDataPreparer,
                 predictor: DefaultAudioEnergyPredictor):
        self._logger = logger
        self._data_preparer = data_preparer
        self._predictor = predictor
        self._separator = self.__class__.__name__

    def predict_for_df(self, df_to_process: pd.DataFrame, model: EnergyModel, target_column: Header) -> pd.DataFrame:
        """
        Processes a DataFrame to calculate energy values.

        This method prepares the data, runs predictions, and then safely merges the
        new energy values back into the original DataFrame, ensuring the index and
        columns are handled correctly.

        Args:
            df_to_process: The input DataFrame containing track data.
            model: The energy model with feature names, scaler, etc.
            target_column: The header for the column where results will be stored.

        Returns:
            A new DataFrame with the calculated energy column added.
        """
        if df_to_process.empty:
            self._logger.info("Input DataFrame is empty, skipping energy calculation.", separator=self._separator)
            return df_to_process

        original_df_with_uuid_index = df_to_process.copy().set_index(Header.UUID.value)

        try:
            prepared_df = self._data_preparer.prepare_for_prediction(
                df_to_process, model.feature_names
            )

            valid_rows = prepared_df[model.feature_names].notna().all(axis=1)
            df_for_prediction = prepared_df[valid_rows]

            if df_for_prediction.empty:
                self._logger.warning(
                    "No valid rows found for energy prediction after data preparation.",
                    separator=self._separator
                )
                if target_column.value not in df_to_process.columns:
                    df_to_process[target_column.value] = float('nan')
                return df_to_process

            result_df = self._predictor.calculate_ratings_for_df(df_for_prediction, target_column)

            result_with_uuid_index = result_df.set_index(Header.UUID.value)
            original_df_with_uuid_index.update(result_with_uuid_index[[target_column.value]])

            return original_df_with_uuid_index.reset_index()

        except KeyError as e:
            self._logger.error(
                f"Critical data issue: DataFrame is missing the required '{Header.UUID.value}' column. Error: {e}",
                separator=self._separator
            )
            return df_to_process
        except Exception as e:
            self._logger.error(
                f"An unexpected error occurred during DataFrame energy calculation: {e}",
                separator=self._separator
            )
            return df_to_process
