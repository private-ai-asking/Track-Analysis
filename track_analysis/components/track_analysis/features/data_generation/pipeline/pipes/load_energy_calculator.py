from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.constants import ENERGY_CALCULATION_REGENERATE_TRACK_INTERVAL
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.configurations.default import \
    DEFAULT_ENERGY_MODEL_CONFIG
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator_factory import \
    EnergyCalculatorFactory, Calculator
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class LoadEnergyCalculator(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.LoadEnergyCalculatorPipe"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

        self._energy_calculator_factory: EnergyCalculatorFactory = EnergyCalculatorFactory(self._logger)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Loading energy calculator if existing...", separator=self._separator)
        data.energy_calculator = self._energy_calculator_factory.get_calculator(Calculator.Default, data.loaded_mfcc_info_cache)
        config = DEFAULT_ENERGY_MODEL_CONFIG
        loaded_model = data.energy_calculator.load(DEFAULT_ENERGY_MODEL_CONFIG)

        valid_model = data.energy_calculator.validate_model(loaded_model, data.loaded_audio_info_cache)

        if valid_model:
            self._logger.info(f"Cached model is still valid, using: '{config.name}'", separator=self._separator)
            data.energy_calculator.set_model(loaded_model)
            return data

        self._logger.info(f"Cached model is invalid, deciding what to do next.", separator=self._separator)

        samples_used_for_training: int = loaded_model.features_shape.training_samples
        current_tracks_in_cache: int = data.loaded_audio_info_cache.shape[0]

        difference = current_tracks_in_cache - samples_used_for_training
        if difference >= ENERGY_CALCULATION_REGENERATE_TRACK_INTERVAL:
            self._logger.info(f"Retraining energy model and recalculating energy ratings because there are {difference} new samples.", separator=self._separator)
            trained_model = data.energy_calculator.train_and_persist(DEFAULT_ENERGY_MODEL_CONFIG, data.loaded_audio_info_cache)
            data.energy_calculator.set_model(trained_model)

            if Header.Energy_Level not in data.headers_to_refill:
                data.headers_to_refill.append(Header.Energy_Level)
        else:
            self._logger.info(f"Not retraining energy model because the difference in tracks is not big enough.", separator=self._separator)
            data.energy_calculator.set_model(loaded_model)

        return data
