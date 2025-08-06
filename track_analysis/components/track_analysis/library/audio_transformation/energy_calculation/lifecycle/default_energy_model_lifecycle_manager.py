from typing import Union

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.lifecycle.energy_lifecycle_manager import \
    EnergyModelLifecycleManager
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.persistence.energy_model_validator import \
    DefaultEnergyModelValidator
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.preprocessing.energy_preparer import \
    EnergyDataPreparer
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.training.trainer import \
    DefaultEnergyModelTrainer
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.utils import \
    get_dataframe_hash


class DefaultEnergyModelLifecycleManager(EnergyModelLifecycleManager):
    """
    Orchestrates the lifecycle of an energy model: training, persistence, and loading.
    """
    def __init__(self,
                 logger: HoornLogger,
                 trainer: DefaultEnergyModelTrainer,
                 persistence: DefaultModelPersistence,
                 validator: DefaultEnergyModelValidator,
                 data_preparer: EnergyDataPreparer):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._trainer = trainer
        self._persistence = persistence
        self._validator = validator
        self._data_preparer = data_preparer

    def load_model(self, config: EnergyModelConfig) -> EnergyModel | None:
        """Loads a model from persistence if it exists and is valid."""
        return self._persistence.load(config)

    def train_and_persist_model(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        """Trains a new model, persists it, and returns it."""
        self._logger.info("Starting energy model training and persistence pipeline.", separator=self._separator)
        prepared_data = self._data_preparer.prepare_for_training(training_data, config)

        model = self._trainer.train(config, prepared_data)
        self._persistence.save(model, config)

        return model

    def train_in_memory(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        """
        Performs the core training pipeline in memory without persistence.
        """
        self._logger.info("Starting in-memory energy model training.", separator=self._separator)
        prepared_data = self._data_preparer.prepare_for_training(training_data, config)

        model = self._trainer.train(config, prepared_data)

        return model

    def validate_model(self, model: EnergyModel | None, data_or_hash: Union[pd.DataFrame, str]) -> bool:
        if model is None:
            return False

        if not isinstance(data_or_hash, str):
            features_df = data_or_hash[model.feature_names].dropna()
            data_hash = get_dataframe_hash(features_df)
        else:
            data_hash = data_or_hash
        return self._validator.is_valid(model, data_hash)
