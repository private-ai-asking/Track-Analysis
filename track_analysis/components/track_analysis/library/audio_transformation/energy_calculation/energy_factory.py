from enum import Enum
from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.default_calculator import \
    DefaultEnergyAlgorithm
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_calculator import \
    EnergyAlgorithm
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.lifecycle.energy_lifecycle_manager import \
    EnergyModelLifecycleManager
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.lifecycle.default_energy_model_lifecycle_manager import \
    DefaultEnergyModelLifecycleManager
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.persistence.energy_model_validator import \
    DefaultEnergyModelValidator
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.persistence.inspection import \
    DefaultInspectionDataPersistence
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.preprocessing.energy_preparer import \
    EnergyDataPreparer
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.processing.energy_model_processor import \
    EnergyModelProcessor
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.training.trainer import \
    DefaultEnergyModelTrainer


class Implementation(Enum):
    """Defines the set of available energy feature implementations."""
    Default = 0

class EnergyFactory:
    """Creates components for the energy feature calculation process."""
    def __init__(self, logger: HoornLogger, cache_dir: Path):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._energy_preparer: EnergyDataPreparer = EnergyDataPreparer(self._logger)
        self._energy_calculator: EnergyModelProcessor = EnergyModelProcessor()
        self._cache_dir = cache_dir

    def create_lifecycle_manager(self, impl: Implementation) -> EnergyModelLifecycleManager | None:
        """Creates an instance of a model lifecycle manager."""
        if impl == Implementation.Default:
            inspection_persistence = DefaultInspectionDataPersistence(self._logger)
            persistence = DefaultModelPersistence(self._logger, self._cache_dir, inspection_persistence)
            trainer = DefaultEnergyModelTrainer(self._logger, persistence, self._energy_calculator)
            validator = DefaultEnergyModelValidator(self._logger)
            return DefaultEnergyModelLifecycleManager(
                self._logger, trainer, persistence, validator, self._energy_preparer
            )

        self._logger.warning(f"Lifecycle Manager for '{impl.name}' not implemented.", separator=self._separator)
        return None

    def create_calculator(self, impl: Implementation, model: EnergyModel) -> EnergyAlgorithm | None:
        """
        Creates an instance of an energy calculator for a specific model.

        Args:
            impl: The implementation type to create.
            model: The trained EnergyModel that the calculator will use for predictions.
        """
        if impl == Implementation.Default:
            energy_preparer = EnergyDataPreparer(self._logger)

            return DefaultEnergyAlgorithm(
                self._logger, self._energy_calculator, energy_preparer, model
            )

        self._logger.warning(f"Calculator for '{impl.name}' not implemented.", separator=self._separator)
        return None
