from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.data_generation.build_csv_pipeline import \
    BuildLibraryDataCSVPipeline, PipelineConfiguration
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel
from track_analysis.components.track_analysis.shared.caching.max_rate_cache import MaxRateCache


class CsvPipelineFactory:
    """Creates the csv pipeline."""
    def __init__(self,
                 logger: HoornLogger,
                 file_handler: FileHandler,
                 string_utils: StringUtils,
                 tag_extractor: TagExtractor,
                 max_rate_cache: MaxRateCache,
                 app_configuration: TrackAnalysisConfigurationModel):
        self._logger = logger
        self._file_handler = file_handler
        self._string_utils = string_utils
        self._tag_extractor = tag_extractor
        self._max_rate_cache = max_rate_cache
        self._app_configuration = app_configuration

    def create(self, num_workers: int, hop_length: int = 512, n_fft: int = 2048) -> BuildLibraryDataCSVPipeline:
        pipeline_config: PipelineConfiguration = PipelineConfiguration(num_workers=num_workers, hop_length=hop_length, n_fft=n_fft)

        pipeline = BuildLibraryDataCSVPipeline(
            logger=self._logger,
            filehandler=self._file_handler, string_utils=self._string_utils,
            tag_extractor=self._tag_extractor, max_rate_cache=self._max_rate_cache,
            configuration=pipeline_config, app_configuration=self._app_configuration
        )

        return pipeline
