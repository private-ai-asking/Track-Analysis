import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.prediction.dataframe_prediction_pipeline import \
    DataFramePredictionPipeline
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.prediction.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.prediction.row_prediction_pipeline import \
    RowPredictionPipeline
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.preprocessing.energy_preparer import \
    EnergyDataPreparer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class DefaultEnergyCalculator(EnergyCalculator):
    """
    Uses a given energy model to calculate energy ratings for tracks.
    Delegates data preparation and prediction to dedicated pipeline components.
    """

    def __init__(self,
                 logger: HoornLogger,
                 predictor: DefaultAudioEnergyPredictor,
                 data_preparer: EnergyDataPreparer,
                 model: EnergyModel):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._energy_model = model
        self._predictor = predictor
        self._predictor.set_model(self._energy_model)

        self._row_pipeline = RowPredictionPipeline(logger, data_preparer, predictor)
        self._df_pipeline = DataFramePredictionPipeline(logger, data_preparer, predictor)

    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        """Calculates energy for a DataFrame of tracks by delegating to the DF pipeline."""
        return self._df_pipeline.predict_for_df(df_to_process, self._energy_model, target_column)

    def calculate_energy_for_row(self, row: pd.Series) -> float:
        """Calculates energy for a single track row."""
        return self._row_pipeline.predict_for_row(row, self._energy_model)
