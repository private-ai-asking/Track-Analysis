from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class EnergyCalculator(ABC):
    """A high-level facade for the interaction with energy calculation."""

    @abstractmethod
    def train_and_persist(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        """
        Orchestrates the full model training and persistence pipeline.

        This method trains a new model based on the provided configuration and data,
        then saves the resulting artifacts to the cache for future use.

        Args:
            config: The configuration object defining the model to be trained.
            training_data: The full dataset used for training.

        Returns:
            The newly trained and persisted EnergyModel object.
        """
        ...

    @abstractmethod
    def train_or_load(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> Tuple[EnergyModel, bool]:
        """
        Performs the core training pipeline in memory without persistence.

        This method executes all training steps (scaling, PCA, curve fitting)
        and returns the resulting model object, but does not save it to the cache.

        Args:
            config: The configuration defining how to train the model.
            training_data: The dataset to train on.

        Returns:
            The trained, in-memory EnergyModel object and whether the model was loaded from cache or freshly generated.
        """
        ...

    @abstractmethod
    def persist(self, config: EnergyModelConfig, model: EnergyModel) -> None:
        """
        Saves a trained model and its inspection data to the cache.

        Args:
            config: The configuration, used to determine the cache path.
            model: The trained EnergyModel object to save.
        """
        ...

    @abstractmethod
    def calculate_energy_for_row(self, row: pd.Series) -> float:
        """
        Calculates the energy score for a single track's data.

        Args:
            row: A pandas Series containing the audio features for one track.

        Returns:
            The calculated energy score as a float.
        """
        ...

    @abstractmethod
    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        """
        Calculates energy ratings for a given DataFrame and returns it with a new/updated column.

        This method is stateless and does not modify the class's internal data. It is the
        recommended method for batch processing.

        Args:
            df_to_process: The pandas DataFrame containing the rows to process.
            target_column: The Header enum for the column to add/update with the ratings.

        Returns:
            A new DataFrame with the calculated energy ratings.
        """
        ...
