import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CURRENT_ENERGY_TRAINING_VERSION_TO_USE, \
    ENERGY_CALCULATION_REGENERATE_LIBRARY_GROWTH_PERC
from track_analysis.components.track_analysis.features.data_generation.helpers.energy_training_config_factory import \
    TrainingConfigFactory
from track_analysis.components.track_analysis.features.data_generation.helpers.energy_training_data_builder import \
    TrainingDataBuilder
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.configuration.default import \
    DEFAULT_ENERGY_MODEL_CONFIG
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_factory import \
    EnergyFactory, Implementation
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.lifecycle.energy_lifecycle_manager import \
    EnergyModelLifecycleManager
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig


class EnergyLoader:
    """
    Helper class for loading the energy model.
    """
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator: str = self.__class__.__name__

        self._energy_factory: EnergyFactory = EnergyFactory(logger)
        self._data_builder: TrainingDataBuilder = TrainingDataBuilder()
        self._config_factory: TrainingConfigFactory = TrainingConfigFactory()

    def get_combined_data(self, main_data_frame: pd.DataFrame, mfcc_data_frame: pd.DataFrame) -> pd.DataFrame:
        base_config = DEFAULT_ENERGY_MODEL_CONFIG
        main_df = self._data_builder.build(
            main_df=main_data_frame,
            mfcc_df=mfcc_data_frame,
            use_mfcc=base_config.use_mfcc
        )

        return main_df

    def get_calculator(self, main_data_frame: pd.DataFrame, mfcc_data_frame: pd.DataFrame) -> EnergyCalculator:
        manager = self._energy_factory.create_lifecycle_manager(Implementation.Default)
        if not manager:
            raise RuntimeError("Failed to create the EnergyModelLifecycleManager.")

        base_config = DEFAULT_ENERGY_MODEL_CONFIG
        main_df = self.get_combined_data(main_data_frame, mfcc_data_frame)

        final_config = self._config_factory.create(base_config, main_df, CURRENT_ENERGY_TRAINING_VERSION_TO_USE)

        loaded_model = manager.load_model(final_config)
        is_model_valid = manager.validate_model(loaded_model, main_df)

        if is_model_valid:
            self._logger.info(f"Cached model is still valid, using: '{final_config.name}' @ '{final_config.version}'", separator=self._separator)
            model_to_use = loaded_model
        else:
            model_to_use = self._handle_invalid_or_missing_model(manager, loaded_model, final_config, main_df)

        return self._energy_factory.create_calculator(Implementation.Default, model_to_use)

    def _handle_invalid_or_missing_model(
            self,
            manager: EnergyModelLifecycleManager,
            loaded_model: EnergyModel | None,
            config: EnergyModelConfig,
            main_df: pd.DataFrame
    ) -> EnergyModel:
        """Decides whether to retrain or use a stale model."""
        self._logger.info("Cached model is invalid or missing, deciding what to do next.", separator=self._separator)

        if not loaded_model:
            self._logger.info("No existing model found. Training a new one.", separator=self._separator)
            return self._train_new_model(manager, main_df, config)

        if self._should_retrain_model(loaded_model, main_df.shape[0]):
            self._train_new_model(manager, main_df, config)

        return loaded_model

    def _should_retrain_model(self, model: EnergyModel, current_track_count: int) -> bool:
        """Determines if the number of new tracks meets the retraining threshold."""
        samples_used_for_training = model.features_shape.training_samples
        growth_percentage = (current_track_count - samples_used_for_training) / samples_used_for_training

        should_retrain = growth_percentage >= ENERGY_CALCULATION_REGENERATE_LIBRARY_GROWTH_PERC
        if should_retrain:
            self._logger.info(f"Retraining model: growth percentage ({growth_percentage}) exceeds threshold\n"
                              f"Using old model for current calculations, because the new model needs to be manually verified..", separator=self._separator)
        else:
            self._logger.info(f"Using current model for current calculations because ({growth_percentage}) is below threshold.", separator=self._separator)

        return should_retrain

    def _train_new_model(self, manager: EnergyModelLifecycleManager, df: pd.DataFrame, config: EnergyModelConfig) -> EnergyModel:
        """Handles the logic for training a new model and updating the context."""
        increased_config = self._config_factory.increase_training_version(config)
        model = manager.train_and_persist_model(increased_config, df)

        return model
