import dataclasses

import pandas as pd

from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig


class TrainingConfigFactory:
    """
    Creates the final, complete EnergyModelConfig for a training run.
    """
    @staticmethod
    def create(base_config: EnergyModelConfig,
               training_df: pd.DataFrame,
               training_version: int) -> EnergyModelConfig:
        """
        Constructs the final config by dynamically adding MFCC feature names if required.

        Args:
            base_config: The base model configuration with core features.
            training_df: The final, merged DataFrame that will be used for training.
            training_version: The version of the training dataset to use.

        Returns:
            A new, immutable EnergyModelConfig with the complete list of feature columns.
        """
        training_version_adjusted = dataclasses.replace(base_config, version=training_version)

        if not training_version_adjusted.use_mfcc:
            return training_version_adjusted

        # Discover MFCC columns directly from the merged training data
        mfcc_columns = [col for col in training_df.columns if col.startswith('mfcc_')]
        mfcc_columns.remove("mfcc_mean_0")
        mfcc_columns.remove("mfcc_std_0")

        # Combine base features with discovered MFCC features
        final_feature_names = training_version_adjusted.get_feature_names() + mfcc_columns

        # Create a new, immutable config object with the complete feature list
        return dataclasses.replace(training_version_adjusted, feature_columns=final_feature_names)

    @staticmethod
    def increase_training_version(base_config: EnergyModelConfig) -> EnergyModelConfig:
        """Increases the model's training version to separate between training runs on the same base config."""
        return dataclasses.replace(base_config, version=base_config.version+1)
