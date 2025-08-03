from typing import overload, Union

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.energy_model_validator import \
    DefaultEnergyModelValidator
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.persistence.model_persistence import \
    DefaultModelPersistence
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.trainer import \
    DefaultEnergyModelTrainer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.utils import \
    get_dataframe_hash
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class DefaultEnergyCalculator(EnergyCalculator):
    def __init__(self, logger: HoornLogger,
                 trainer: DefaultEnergyModelTrainer,
                 predictor: DefaultAudioEnergyPredictor,
                 persistence: DefaultModelPersistence,
                 validator: DefaultEnergyModelValidator,
                 ):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._trainer = trainer
        self._predictor = predictor
        self._persistence = persistence
        self._validator = validator

    def set_model(self, model: EnergyModel) -> None:
        self._predictor.set_model(model)

    @overload
    def validate_model(self, model: EnergyModel | None, current_data: pd.DataFrame) -> bool:
        ...

    @overload
    def validate_model(self, model: EnergyModel | None, current_data_hash: str) -> bool:
        ...

    def validate_model(self, model: EnergyModel | None, data_or_hash: Union[pd.DataFrame, str]) -> bool:
        if not isinstance(data_or_hash, str):
            data_hash = get_dataframe_hash(data_or_hash)
        else:
            data_hash = data_or_hash

        self._validator.is_valid(model, data_hash)

    def load(self, config: EnergyModelConfig) -> EnergyModel | None:
        return self._persistence.load(config)

    def train_and_persist(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        model = self.train(config, training_data)
        self.persist(config, model)

        return model

    def train(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        return self._trainer.train(config, training_data)

    def persist(self, config: EnergyModelConfig, model: EnergyModel) -> None:
        self._persistence.save(model, config)

    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        return self._predictor.calculate_ratings_for_df(df_to_process, target_column)

    def calculate_energy_for_row(self, row: pd.Series) -> float:
        return self._predictor.calculate_energy_for_row(row)
