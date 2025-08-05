from typing import Union

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.configuration.configuration_resolver import \
    DefaultEnergyConfigResolver
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.energy_model_validator import \
    DefaultEnergyModelValidator
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.preprocessing.energy_preparer import \
    EnergyDataPreparer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.training.trainer import \
    DefaultEnergyModelTrainer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.utils import \
    get_dataframe_hash
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_lifecycle_manager import \
    EnergyModelLifecycleManager


class DefaultEnergyModelLifecycleManager(EnergyModelLifecycleManager):
    """
    Orchestrates the lifecycle of an energy model: training, persistence, and loading.
    """
    def __init__(self,
                 logger: HoornLogger,
                 trainer: DefaultEnergyModelTrainer,
                 persistence: DefaultModelPersistence,
                 validator: DefaultEnergyModelValidator,
                 data_preparer: EnergyDataPreparer,
                 config_resolver: DefaultEnergyConfigResolver):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._trainer = trainer
        self._persistence = persistence
        self._validator = validator
        self._data_preparer = data_preparer
        self._config_resolver = config_resolver

    def load_model(self, config: EnergyModelConfig) -> EnergyModel | None:
        """Loads a model from persistence if it exists and is valid."""
        final_config = self._config_resolver.resolve(config)
        return self._persistence.load(final_config)

    def train_and_persist_model(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        """Trains a new model, persists it, and returns it."""
        self._logger.info("Starting energy model training and persistence pipeline.", separator=self._separator)
        final_config = self._config_resolver.resolve(config)
        prepared_data = self._data_preparer.prepare_for_training(training_data, final_config)

        model = self._trainer.train(final_config, prepared_data)

        persistence_config = EnergyModelConfig(
            name=final_config.name,
            feature_columns=model.feature_names,
            use_mfcc=any("mfcc_" in f for f in model.feature_names)
        )
        self._persistence.save(model, persistence_config)

        return model

    def train_in_memory(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        """
        Performs the core training pipeline in memory without persistence.
        """
        self._logger.info("Starting in-memory energy model training.", separator=self._separator)
        final_config = self._config_resolver.resolve(config)

        prepared_data = self._data_preparer.prepare_for_training(training_data, final_config)

        model = self._trainer.train(final_config, prepared_data)

        return model

    def validate_model(self, model: EnergyModel | None, data_or_hash: Union[pd.DataFrame, str]) -> bool:
        if not isinstance(data_or_hash, str):
            data_hash = get_dataframe_hash(data_or_hash)
        else:
            data_hash = data_or_hash
        return self._validator.is_valid(model, data_hash)
