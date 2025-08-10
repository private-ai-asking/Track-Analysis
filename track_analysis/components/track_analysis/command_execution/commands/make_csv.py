from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.data_generation.factory.csv_pipeline_factory import \
    CsvPipelineFactory
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class MakeCSVCommand(CommandExecutionModel):
    def __init__(self,
                 logger: HoornLogger,
                 app_configuration: TrackAnalysisConfigurationModel,
                 csv_pipeline_factory: CsvPipelineFactory,
                 ):
        super().__init__(logger)
        self._app_configuration: TrackAnalysisConfigurationModel = app_configuration
        self._csv_pipeline_factory: CsvPipelineFactory = csv_pipeline_factory

    def execute(self, profile: T) -> P:
        key_progression_path: Path = self._app_configuration.paths.key_progression_data
        pipeline_context = LibraryDataGenerationPipelineContext(
            source_dir=self._app_configuration.paths.root_music_library,
            main_data_output_file_path=self._app_configuration.paths.library_data,
            key_progression_output_file_path=key_progression_path,
            mfcc_data_output_file_path=self._app_configuration.paths.mfcc_data,
            use_threads=True,
            max_new_tracks_per_run=self._app_configuration.additional_config.max_new_tracks_per_run,
            missing_headers_to_fill=[
                Header.Onset_Rate_Variation,
                Header.High_Ratio,
                Header.Mid_Ratio,
                Header.Low_Mid_Ratio,
                Header.Bass_Ratio,
                Header.Sub_Bass_Ratio,
                Header.Spectral_Bandwidth_STD,
                Header.Spectral_Bandwidth_Mean,
                Header.Zero_Crossing_Rate_STD,
                Header.Spectral_Entropy,
                Header.Spectral_Kurtosis,
                Header.Spectral_Skewness,
                Header.Spectral_Contrast_STD,
                Header.Spectral_Flatness_STD,
                Header.Spectral_Flux_Std,
                Header.Spectral_Centroid_Std,
                Header.Chroma_Entropy,
                Header.HPR,
                Header.Rhythmic_Regularity,
                Header.Beat_Strength,
                Header.Integrated_LUFS_Range,
                Header.Integrated_LUFS_STD,
                Header.Integrated_LUFS_Mean
            ],
            headers_to_refill=[
                Header.MFCC,
                # Header.Energy_Level
            ],
            end_at_energy_calculation_loading=False
        )

        pipeline = self._csv_pipeline_factory.create(num_workers=self._app_configuration.additional_config.num_workers_cpu_heavy - 10)
        pipeline.build_pipeline()
        pipeline.flow(pipeline_context)

        self._logger.info("CSV has been successfully created.")

    @property
    def default_command_keys(self) -> List[str]:
        return ["make_csv", "mc"]

    @property
    def command_description(self) -> str:
        return "Exports the library data into CSV format."

