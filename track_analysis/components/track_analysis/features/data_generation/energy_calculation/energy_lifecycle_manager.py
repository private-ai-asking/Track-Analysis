from abc import ABC, abstractmethod
from typing import overload, Union

import pandas as pd

from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig


class EnergyModelLifecycleManager(ABC):
    """
    An interface for managing the lifecycle of an energy model, including
    training, persistence, validation, and loading.
    """

    @abstractmethod
    def load_model(self, config: EnergyModelConfig) -> EnergyModel | None:
        """
        Loads a trained model from persistence based on the configuration.

        Args:
            config: The configuration defining the model to load.

        Returns:
            The loaded EnergyModel object, or None if not found or invalid.
        """
        ...

    @abstractmethod
    def train_and_persist_model(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        """
        Orchestrates the full model training and persistence pipeline.

        Args:
            config: The configuration object defining the model to be trained.
            training_data: The full dataset used for training.

        Returns:
            The newly trained and persisted EnergyModel object.
        """
        ...

    @abstractmethod
    def train_in_memory(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        """
        Performs the core training pipeline in memory without persistence.

        Args:
            config: The configuration defining how to train the model.
            training_data: The dataset to train on.

        Returns:
            The trained, in-memory EnergyModel object.
        """
        ...

    @overload
    def validate_model(self, model: EnergyModel | None, current_data: pd.DataFrame) -> bool: ...

    @overload
    def validate_model(self, model: EnergyModel | None, current_data_hash: str) -> bool: ...

    @abstractmethod
    def validate_model(self, model: EnergyModel | None, data_or_hash: Union[pd.DataFrame, str]) -> bool:
        """
        Checks if a loaded model is valid against the current dataset.

        Args:
            model: The loaded EnergyModel object to validate, or None.
            data_or_hash: The current data as a DataFrame to be hashed, or a pre-computed hash string.

        Returns:
            True if the model is valid and can be used, False otherwise.
        """
        ...
