from typing import List

import pandas as pd


from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig


class DefaultEnergyConfigResolver:
    """
    Resolves a base EnergyModelConfig into a final config by adding MFCC features.
    """
    def __init__(self, mfcc_data: pd.DataFrame):
        if mfcc_data is None or mfcc_data.empty:
            raise ValueError("MFCC data must be provided to the DefaultEnergyConfigResolver.")
        self._mfcc_data = mfcc_data

    def resolve(self, base_config: EnergyModelConfig) -> EnergyModelConfig:
        """
        Constructs the final model config by adding MFCC feature names if required.
        """
        final_feature_names = base_config.get_feature_names()

        if base_config.use_mfcc:
            mfcc_df_to_process = self._mfcc_data.drop(['mfcc_mean_0', 'mfcc_std_0'], axis=1, errors='ignore')
            mfcc_feature_names = [col for col in mfcc_df_to_process.columns if col != Header.UUID.value]
            final_feature_names.extend(mfcc_feature_names)

        return self._create_final_config(base_config, final_feature_names)

    @staticmethod
    def _create_final_config(initial_config: EnergyModelConfig, final_feature_names: List[str]) -> EnergyModelConfig:
        """Creates a new config object based on the final feature list."""
        return EnergyModelConfig(
            name=initial_config.name,
            feature_columns=final_feature_names,
            use_mfcc=any("mfcc_" in f for f in final_feature_names)
        )
