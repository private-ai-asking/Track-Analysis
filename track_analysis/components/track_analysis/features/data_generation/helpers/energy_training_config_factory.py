import dataclasses

import pandas as pd

from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig


class TrainingConfigFactory:
    """
    Creates the final, complete EnergyModelConfig for a training run.
    """
    def create(self,
               base_config: EnergyModelConfig,
               training_df: pd.DataFrame) -> EnergyModelConfig:
        """
        Constructs the final config by dynamically adding MFCC feature names if required.

        Args:
            base_config: The base model configuration with core features.
            training_df: The final, merged DataFrame that will be used for training.

        Returns:
            A new, immutable EnergyModelConfig with the complete list of feature columns.
        """
        if not base_config.use_mfcc:
            return base_config

        # Discover MFCC columns directly from the merged training data
        mfcc_columns = [col for col in training_df.columns if col.startswith('mfcc_')]
        mfcc_columns.remove("mfcc_mean_0")
        mfcc_columns.remove("mfcc_std_0")

        # Combine base features with discovered MFCC features
        final_feature_names = base_config.get_feature_names() + mfcc_columns

        # Create a new, immutable config object with the complete feature list
        return dataclasses.replace(base_config, feature_columns=final_feature_names)
