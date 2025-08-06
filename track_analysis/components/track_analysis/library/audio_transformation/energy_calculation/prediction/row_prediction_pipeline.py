import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.prediction.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.preprocessing.energy_preparer import \
    EnergyDataPreparer


class RowPredictionPipeline:
    """
    Handles the preparation, validation, and prediction for a single row of data.
    """
    def __init__(self,
                 logger: HoornLogger,
                 data_preparer: EnergyDataPreparer,
                 predictor: DefaultAudioEnergyPredictor):
        self._logger = logger
        self._data_preparer = data_preparer
        self._predictor = predictor
        self._separator = self.__class__.__name__

    def predict_for_row(self, row: pd.Series, model: EnergyModel) -> float:
        """
        Processes a single row to calculate its energy value. Returns NaN on failure.
        """
        row_uuid = row.get(Header.UUID.value, "N/A")
        try:
            # 1. Prepare
            single_row_df = row.to_frame().T
            prepared_df = self._data_preparer.prepare_for_prediction(
                single_row_df, model.feature_names
            )

            # 2. Validate
            if prepared_df[model.feature_names].isnull().values.any():
                raise ValueError("Incomplete feature data after preparation (e.g., missing MFCCs).")

            # 3. Predict
            prepared_row = prepared_df.iloc[0]
            return self._predictor.calculate_energy_for_row(prepared_row)

        except (KeyError, ValueError) as e:
            self._logger.warning(
                f"Could not calculate energy for track UUID {row_uuid} due to data issue: {e}. Returning NaN.",
                separator=self._separator
            )
            return float('nan')
        except Exception as e:
            self._logger.error(
                f"An unexpected error occurred for track UUID {row_uuid}: {e}",
                separator=self._separator
            )
            return float('nan')
