from enum import Enum

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.default_calculator import \
    DefaultEnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.energy_model_validator import \
    DefaultEnergyModelValidator
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.inspection import \
    DefaultInspectionDataPersistence
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.trainer import \
    DefaultEnergyModelTrainer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator


class Calculator(Enum):
    Default = 0

class EnergyCalculatorFactory:
    """Creates the energy calculator instances."""
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

    def get_calculator(self, calculator: Calculator, mfcc_data: pd.DataFrame = pd.DataFrame()) -> EnergyCalculator | None:
        if calculator == Calculator.Default:
            inspection_persistence = DefaultInspectionDataPersistence(self._logger)
            persistence = DefaultModelPersistence(self._logger, CACHE_DIRECTORY, inspection_persistence)
            trainer = DefaultEnergyModelTrainer(self._logger, persistence)
            predictor = DefaultAudioEnergyPredictor(self._logger)
            validator = DefaultEnergyModelValidator(self._logger)
            return DefaultEnergyCalculator(self._logger, trainer, predictor, persistence, validator, mfcc_data)

        self._logger.warning(f"Calculator {calculator.name} not implemented.", separator=self._separator)
        return None
