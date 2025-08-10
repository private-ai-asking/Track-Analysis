from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.helpers.energy_loader import EnergyLoader
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class LoadEnergyCalculator(IPipe):
    """
    A pipeline step that loads, validates, or retrains an energy model and attaches
    a ready-to-use EnergyCalculator to the pipeline context.
    """
    def __init__(self, logger: HoornLogger, track_analysis_configuration: TrackAnalysisConfigurationModel):
        self._separator = "BuildCSV.LoadEnergyCalculatorPipe"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)
        self._energy_loader: EnergyLoader = EnergyLoader(logger,
                                                         energy_training_version_to_use=track_analysis_configuration.additional_config.current_energy_training_version_to_use,
                                                         regenerate_library_growth_threshold=track_analysis_configuration.additional_config.energy_calculation_regenerate_library_growth_perc,
                                                         cache_dir=track_analysis_configuration.paths.cache_dir)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        """Orchestrates the model selection and calculator creation."""
        self._logger.trace("Determining which energy model to use...", separator=self._separator)

        data.energy_calculator = self._energy_loader.get_calculator(data.loaded_audio_info_cache, data.loaded_mfcc_info_cache)
        self._logger.info("Energy calculator has been successfully configured and attached to the pipeline.", separator=self._separator)

        return data
