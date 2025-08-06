from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.constants import ENERGY_CALCULATION_REGENERATE_TRACK_INTERVAL
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

        factory = EnergyFactory(self._logger, data.loaded_mfcc_info_cache)
        manager = factory.create_lifecycle_manager(Implementation.Default)
        if not manager:
            raise RuntimeError("Failed to create the EnergyModelLifecycleManager.")

        loaded_model = manager.load_model(DEFAULT_ENERGY_MODEL_CONFIG)
        is_model_valid = manager.validate_model(loaded_model, data.loaded_audio_info_cache)

        if is_model_valid:
            self._logger.info(f"Cached model is still valid, using: '{DEFAULT_ENERGY_MODEL_CONFIG.name}'", separator=self._separator)
            model_to_use = loaded_model
        else:
            model_to_use = self._handle_invalid_or_missing_model(manager, loaded_model, data)

        self._attach_calculator_to_context(factory, model_to_use, data)

        return data

    def _handle_invalid_or_missing_model(
            self,
            manager: EnergyModelLifecycleManager,
            loaded_model: EnergyModel | None,
            data: LibraryDataGenerationPipelineContext
    ) -> EnergyModel:
        """Decides whether to retrain or use a stale model when the current one is invalid or missing."""
        self._logger.info("Cached model is invalid or missing, deciding what to do next.", separator=self._separator)

        if not loaded_model:
            self._logger.info("No existing model found. Training a new one.", separator=self._separator)
            return self._train_new_model(manager, data)

        if self._should_retrain_model(loaded_model, data.loaded_audio_info_cache.shape[0]):
            return self._train_new_model(manager, data)
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
    def _train_new_model(manager: EnergyModelLifecycleManager, data: LibraryDataGenerationPipelineContext) -> EnergyModel:
        """Handles the logic for training a new model and updating the context."""
        model = manager.train_and_persist_model(DEFAULT_ENERGY_MODEL_CONFIG, data.loaded_audio_info_cache)

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
