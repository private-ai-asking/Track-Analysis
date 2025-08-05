from enum import Enum

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.configuration.configuration_resolver import \
    DefaultEnergyConfigResolver
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.default_calculator import \
    DefaultEnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.lifecycle.default_energy_model_lifecycle_manager import \
    DefaultEnergyModelLifecycleManager
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.prediction.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.preprocessing.energy_preparer import \
    EnergyDataPreparer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.energy_model_validator import \
    DefaultEnergyModelValidator
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.inspection import \
    DefaultInspectionDataPersistence
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.training.trainer import \
    DefaultEnergyModelTrainer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_lifecycle_manager import \
    EnergyModelLifecycleManager


class Implementation(Enum):
    """Defines the set of available energy feature implementations."""
    Default = 0

class EnergyFactory:
    """Creates components for the energy feature calculation process."""
    def __init__(self, logger: HoornLogger, mfcc_data: pd.DataFrame = pd.DataFrame()):
        self._logger = logger
        self._mfcc_data = mfcc_data
        self._separator = self.__class__.__name__

    def create_lifecycle_manager(self, impl: Implementation) -> EnergyModelLifecycleManager | None:
        """Creates an instance of a model lifecycle manager."""
        if impl == Implementation.Default:
            inspection_persistence = DefaultInspectionDataPersistence(self._logger)
            persistence = DefaultModelPersistence(self._logger, CACHE_DIRECTORY, inspection_persistence)
            trainer = DefaultEnergyModelTrainer(self._logger, persistence)
            validator = DefaultEnergyModelValidator(self._logger)
            energy_preparer = EnergyDataPreparer(self._mfcc_data, self._logger)
            config_resolver = DefaultEnergyConfigResolver(self._mfcc_data)

            return DefaultEnergyModelLifecycleManager(
                self._logger, trainer, persistence, validator, energy_preparer, config_resolver
            )

        self._logger.warning(f"Lifecycle Manager for '{impl.name}' not implemented.", separator=self._separator)
        return None

    def create_calculator(self, impl: Implementation, model: EnergyModel) -> EnergyCalculator | None:
        """
        Creates an instance of an energy calculator for a specific model.

        Args:
            impl: The implementation type to create.
            model: The trained EnergyModel that the calculator will use for predictions.
        """
        if impl == Implementation.Default:
            predictor = DefaultAudioEnergyPredictor(self._logger)
            energy_preparer = EnergyDataPreparer(self._mfcc_data, self._logger)

            return DefaultEnergyCalculator(
                self._logger, predictor, energy_preparer, model
            )

        self._logger.warning(f"Calculator for '{impl.name}' not implemented.", separator=self._separator)
        return None
