import pprint
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_calculator import \
    EnergyAlgorithm
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING


class EnergyProvider(AudioDataFeatureProvider):
    """
    Calculates the energy level of a track by delegating to a configured EnergyCalculator.
    """

    def __init__(self, energy_calculator: EnergyAlgorithm, logger: HoornLogger):
        self._energy_calculator = energy_calculator
        self._feature_to_header_mapping: Dict[str, str] = {
            feature.name: header.value
            for feature, header in FEATURE_TO_HEADER_MAPPING.items()
        }

        self._logger = logger
        self._separator = self.__class__.__name__
        self._logger.debug(f"Mapping:\n{pprint.pformat(self._feature_to_header_mapping)}", separator=self._separator)


    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return self._energy_calculator.get_dependencies()

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ENERGY_LEVEL

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        """
        Passes the dependency data directly to the energy calculator without transformation.
        """
        transformed_data = self._transform_data_into_df(data)
        energy_level = self._energy_calculator.calculate_energy_for_row(transformed_data)

        return {
            AudioDataFeature.ENERGY_LEVEL: energy_level
        }

    def _transform_data_into_df(self, data: Dict[AudioDataFeature, Any]) -> pd.DataFrame:
        # TODO - Remove storage detail dependency
        mfcc_data = {
            self._feature_to_header_mapping[k.name]: v
            for k, v in data.items()
            if k.name in self._feature_to_header_mapping
        }

        mfcc_means_array = np.array(data[AudioDataFeature.MFCC_MEANS])
        mfcc_stds_array = np.array(data[AudioDataFeature.MFCC_STDS])

        mfcc_means_stacked = mfcc_means_array.reshape(1, -1)
        mfcc_stds_stacked = mfcc_stds_array.reshape(1, -1)

        for i in range(mfcc_means_stacked.shape[1]):
            mfcc_data[f"mfcc_mean_{i}"] = mfcc_means_stacked[:, i]
            mfcc_data[f"mfcc_std_{i}"] = mfcc_stds_stacked[:, i]

        final_df = pd.DataFrame(mfcc_data, index=[0])

        return final_df

