import dataclasses

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.constants import ENERGY_CALCULATION_REGENERATE_TRACK_INTERVAL
from track_analysis.components.track_analysis.features.data_generation.helpers.energy_training_config_factory import \
    TrainingConfigFactory
from track_analysis.components.track_analysis.features.data_generation.helpers.energy_training_data_builder import \
    TrainingDataBuilder
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.configuration.default import \
    DEFAULT_ENERGY_MODEL_CONFIG
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_factory import \
    Implementation, EnergyFactory
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_lifecycle_manager import \
    EnergyModelLifecycleManager
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig


class LoadEnergyCalculator(IPipe):
    """
    A pipeline step that loads, validates, or retrains an energy model and attaches
    a ready-to-use EnergyCalculator to the pipeline context.
    """
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.LoadEnergyCalculatorPipe"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        """Orchestrates the model selection and calculator creation."""
        self._logger.trace("Determining which energy model to use...", separator=self._separator)

        factory = EnergyFactory(self._logger)
        manager = factory.create_lifecycle_manager(Implementation.Default)
        if not manager:
            raise RuntimeError("Failed to create the EnergyModelLifecycleManager.")

        data_builder = TrainingDataBuilder()
        config_factory = TrainingConfigFactory()

        base_config = DEFAULT_ENERGY_MODEL_CONFIG

        main_df = data_builder.build(
            main_df=data.loaded_audio_info_cache,
            mfcc_df=data.loaded_mfcc_info_cache,
            use_mfcc=base_config.use_mfcc
        )

        # 2. Create the final configuration using the factory
        final_config = config_factory.create(base_config, main_df)

        loaded_model = manager.load_model(final_config)
        is_model_valid = manager.validate_model(loaded_model, main_df)

        if is_model_valid:
            self._logger.info(f"Cached model is still valid, using: '{final_config.name}'", separator=self._separator)
            model_to_use = loaded_model
        else:
            model_to_use = self._handle_invalid_or_missing_model(manager, loaded_model, data, final_config, main_df)

        self._attach_calculator_to_context(factory, model_to_use, data)

        return data

    def _handle_invalid_or_missing_model(
            self,
            manager: EnergyModelLifecycleManager,
            loaded_model: EnergyModel | None,
            data: LibraryDataGenerationPipelineContext,
            config: EnergyModelConfig,
            df: pd.DataFrame
    ) -> EnergyModel:
        """Decides whether to retrain or use a stale model."""
        self._logger.info("Cached model is invalid or missing, deciding what to do next.", separator=self._separator)

        if not loaded_model:
            self._logger.info("No existing model found. Training a new one.", separator=self._separator)
            return self._train_new_model(manager, data, df, config)

        if self._should_retrain_model(loaded_model, data.loaded_audio_info_cache.shape[0]):
            return self._train_new_model(manager, data, df, config)
        else:
            self._logger.info("Using stale model: new track count is below retraining threshold.", separator=self._separator)
            return loaded_model

    def _should_retrain_model(self, model: EnergyModel, current_track_count: int) -> bool:
        """Determines if the number of new tracks meets the retraining threshold."""
        samples_used_for_training = model.features_shape.training_samples
        difference = current_track_count - samples_used_for_training

        should_retrain = difference >= ENERGY_CALCULATION_REGENERATE_TRACK_INTERVAL
        if should_retrain:
            self._logger.info(f"Retraining model: new track count ({difference}) exceeds threshold.", separator=self._separator)

        return should_retrain

    @staticmethod
    def _train_new_model(manager: EnergyModelLifecycleManager, data: LibraryDataGenerationPipelineContext, df: pd.DataFrame, config: EnergyModelConfig) -> EnergyModel:
        """Handles the logic for training a new model and updating the context."""
        model = manager.train_and_persist_model(config, df)

        if Header.Energy_Level not in data.headers_to_refill:
            data.headers_to_refill.append(Header.Energy_Level)

        return model

    def _attach_calculator_to_context(
            self,
            factory: EnergyFactory,
            model_to_use: EnergyModel | None,
            data: LibraryDataGenerationPipelineContext
    ) -> None:
        """Creates the final calculator and attaches it to the pipeline context."""
        if model_to_use:
            data.energy_calculator = factory.create_calculator(Implementation.Default, model_to_use)
            self._logger.info("Energy calculator has been successfully configured and attached to the pipeline.", separator=self._separator)
        else:
            raise RuntimeError("Could not configure the energy calculator as no model could be loaded or trained.")
