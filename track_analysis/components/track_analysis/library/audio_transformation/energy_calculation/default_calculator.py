import pprint
from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.prediction.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.preprocessing.energy_preparer import \
    EnergyDataPreparer
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING


class DefaultEnergyCalculator(EnergyCalculator):
    """
    Uses a given energy model to calculate energy ratings for tracks.
    Delegates data preparation and prediction to dedicated components.
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
        self._data_preparer = data_preparer

        self._header_to_feature_mapping = {v.value: k for k, v in FEATURE_TO_HEADER_MAPPING.items()}

    def get_dependencies(self) -> List[AudioDataFeature]:
        deps = []

        for header_name in self._energy_model.feature_names:
            if header_name in self._header_to_feature_mapping:
                deps.append(self._header_to_feature_mapping[header_name])
            elif "mfcc_" in header_name:
                self._handle_mfcc_header(deps)

        return list(set(deps))

    @staticmethod
    def _handle_mfcc_header(dependencies: List[AudioDataFeature]):
        if AudioDataFeature.MFCC_MEANS not in dependencies:
            dependencies.append(AudioDataFeature.MFCC_MEANS)
        if AudioDataFeature.MFCC_STDS not in dependencies:
            dependencies.append(AudioDataFeature.MFCC_STDS)

    def calculate_energy_for_row(self, row_df: pd.DataFrame) -> float:
        """Calculates energy for a single track provided as a one-row DataFrame."""
        try:
            prepared_df = self._data_preparer.prepare_for_prediction(
                row_df, self._energy_model.feature_names
            )
            new_row = prepared_df.iloc[0]

            return self._predictor.calculate_energy_for_row(new_row)

        except (KeyError, ValueError) as e:
            row_uuid = row_df[Header.UUID.value].iloc[0] if Header.UUID.value in row_df else "N/A"
            self._logger.warning(
                f"Could not calculate energy for track UUID {row_uuid} due to data issue: {e}. Returning NaN.",
                separator=self._separator
            )
            return float('nan')
        except Exception as e:
            self._logger.error(
                f"An unexpected error occurred during energy calculation: {e}",
                separator=self._separator
            )
            return float('nan')
