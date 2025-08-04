import pprint
from typing import Union, List, Tuple

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
    def __init__(self,
                 logger: HoornLogger,
                 trainer: DefaultEnergyModelTrainer,
                 predictor: DefaultAudioEnergyPredictor,
                 persistence: DefaultModelPersistence,
                 validator: DefaultEnergyModelValidator,
                 mfcc_data: pd.DataFrame,
                 ):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._trainer = trainer
        self._predictor = predictor
        self._persistence = persistence
        self._validator = validator
        self._mfcc_data = mfcc_data

        self._energy_model: EnergyModel | None = None
        self._model_config: EnergyModelConfig | None = None

    @staticmethod
    def _get_final_config(initial_config: EnergyModelConfig, final_feature_names: List[str]) -> EnergyModelConfig:
        """Creates a new config object based on the final feature list."""
        return EnergyModelConfig(
            name=initial_config.name,
            feature_columns=final_feature_names,
            use_mfcc=any("mfcc_" in f for f in final_feature_names)
        )

    def _merge_with_mfcc_data(self, main_df: pd.DataFrame, mfcc_columns_to_join: List[str]) -> pd.DataFrame:
        if self._mfcc_data is None or self._mfcc_data.empty:
            raise ValueError("MFCC data was not provided or was empty.")

        mfcc_df_indexed = self._mfcc_data.set_index(Header.UUID.value)

        available_mfcc_columns = [col for col in mfcc_columns_to_join if col in mfcc_df_indexed.columns]
        missing_columns = set(mfcc_columns_to_join) - set(available_mfcc_columns)

        if missing_columns:
            self._logger.warning(
                f"Missing MFCC columns: {pprint.pformat(missing_columns)}. "
                "These features will be treated as NaN.",
                separator=self._separator
            )

        mfcc_df_filtered = mfcc_df_indexed.loc[:, available_mfcc_columns]

        main_df_indexed = main_df.set_index(Header.UUID.value)
        merged_df = main_df_indexed.join(mfcc_df_filtered, how='inner')

        for col in missing_columns:
            merged_df[col] = pd.NA

        return merged_df.reset_index()

    def _prepare_training_data_and_config(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, EnergyModelConfig]:
        final_training_data = training_data.copy()

        # Start with the base features from the initial config
        feature_names = config.get_feature_names()

        if config.use_mfcc:
            mfcc_df_to_process = self._mfcc_data.drop(['mfcc_mean_0', 'mfcc_std_0'], axis=1, errors='ignore')
            mfcc_feature_names = [col for col in mfcc_df_to_process.columns if col != Header.UUID.value]

            final_training_data = self._merge_with_mfcc_data(
                main_df=final_training_data,
                mfcc_columns_to_join=mfcc_feature_names
            )
            feature_names.extend(mfcc_feature_names)

        final_config = self._get_final_config(config, config.get_feature_names())

        return final_training_data, final_config

    def _prepare_prediction_data(self, df_to_process: pd.DataFrame) -> pd.DataFrame:
        if self._energy_model is None:
            raise Exception("Model is not set. Cannot prepare data for prediction.")

        model_features = self._energy_model.feature_names
        mfcc_columns_to_join = [f for f in model_features if f.startswith("mfcc_")]

        if not mfcc_columns_to_join:
            return df_to_process

        prepared_df = self._merge_with_mfcc_data(
            main_df=df_to_process,
            mfcc_columns_to_join=mfcc_columns_to_join
        )

        # IMPORTANT: Ensure the column order matches the model's feature names
        return prepared_df[model_features]

    def set_model(self, model: EnergyModel) -> None:
        self._energy_model = model
        self._predictor.set_model(model)

    def validate_model(self, model: EnergyModel | None, data_or_hash: Union[pd.DataFrame, str]) -> bool:
        if not isinstance(data_or_hash, str):
            data_hash = get_dataframe_hash(data_or_hash)
        else:
            data_hash = data_or_hash

        return self._validator.is_valid(model, data_hash)

    def load(self, config: EnergyModelConfig) -> EnergyModel | None:
        loaded_model = self._persistence.load(config)
        if loaded_model is None:
            return None

        # The loaded model's feature names are the truth.
        self.set_model(loaded_model)

        return loaded_model

    def train_and_persist(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        self._logger.info("Starting energy model training and persistence pipeline.", separator=self._separator)

        prepared_data, final_config = self._prepare_training_data_and_config(config, training_data)

        model = self._trainer.train(final_config, prepared_data)

        self.persist(final_config, model)
        self.set_model(model)

        return model

    def train(self, config: EnergyModelConfig, training_data: pd.DataFrame) -> EnergyModel:
        self._logger.info("Starting in-memory energy model training.", separator=self._separator)

        prepared_data, final_config = self._prepare_training_data_and_config(config, training_data)

        model = self._trainer.train(final_config, prepared_data)
        self.set_model(model)

        return model

    def persist(self, config: EnergyModelConfig, model: EnergyModel) -> None:
        # We need to save with the final config, so we must generate it here.
        final_config = self._get_final_config(config, model.feature_names)
        self._persistence.save(model, final_config)

    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        prepared_df = self._prepare_prediction_data(df_to_process)
        result_df = self._predictor.calculate_ratings_for_df(prepared_df, target_column)

        final_df = df_to_process.copy()

        final_df = final_df.set_index(Header.UUID.value)
        result_df = result_df.set_index(Header.UUID.value)

        final_df[target_column.value] = result_df[target_column.value]

        return final_df.reset_index()

    def calculate_energy_for_row(self, row: pd.Series) -> float:
        if self._energy_model is None:
            raise Exception("Model is not set. Cannot calculate energy.")

        row_uuid = row[Header.UUID.value]
        single_row_df = row.to_frame().T

        try:
            prepared_df = self._prepare_prediction_data(single_row_df)

            if prepared_df.empty:
                self._logger.warning(
                    f"Failed to find a matching UUID in MFCC data for track UUID: {row_uuid}. Returning NaN.",
                    separator=self._separator
                )
                return float('nan')

            prepared_row = prepared_df.iloc[0]
            return self._predictor.calculate_energy_for_row(prepared_row)

        except KeyError as e:
            self._logger.warning(
                f"Missing required feature for track UUID: {row_uuid}. Error: {e}. Returning NaN.",
                separator=self._separator
            )
            return float('nan')
        except Exception as e:
            self._logger.error(
                f"An unexpected error occurred during single-row energy calculation for track UUID: {row_uuid}. Error: {e}",
                separator=self._separator
            )
            return float('nan')
