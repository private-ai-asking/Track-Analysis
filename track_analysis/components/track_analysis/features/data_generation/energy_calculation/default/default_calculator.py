from typing import Tuple

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.trainer import \
    DefaultEnergyModelTrainer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class DefaultEnergyCalculator(EnergyCalculator):
    def __init__(self, logger: HoornLogger,
                 trainer: DefaultEnergyModelTrainer,
                 predictor: DefaultAudioEnergyPredictor,
                 persistence: DefaultModelPersistence,
                 ):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._trainer = trainer
        self._predictor = predictor
        self._persistence = persistence

    def train_and_persist(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        model, loaded = self.train_or_load(config, training_data)

        if not loaded:
            self.persist(config, model)

        return model

    def train_or_load(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> Tuple[EnergyModel, bool]:
        model, loaded = self._trainer.train_or_load(config, training_data)
        self._predictor.set_model(model)
        return model, loaded

    def persist(self, config: EnergyModelConfig, model: EnergyModel) -> None:
        self._persistence.save(model, config)

    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        return self._predictor.calculate_ratings_for_df(df_to_process, target_column)

    def calculate_energy_for_row(self, row: pd.Series) -> float:
        return self._predictor.calculate_energy_for_row(row)
