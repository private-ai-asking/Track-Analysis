import dataclasses

from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.data_generation.builders.key_data_frames_builder import \
    KeyDataFramesBuilder
from track_analysis.components.track_analysis.features.data_generation.builders.metadata_df_builder import \
    MetadataDFBuilder
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING
from track_analysis.components.track_analysis.features.data_generation.mappers.results_mapper import ResultsMapper
from track_analysis.components.track_analysis.shared.caching.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.pipes.batch_process_new_tracks import \
    BatchProcessNewTracks
from track_analysis.components.track_analysis.features.data_generation.pipes.filter_cache import FilterCache
from track_analysis.components.track_analysis.features.data_generation.pipes.filter_files import FilterFiles
from track_analysis.components.track_analysis.features.data_generation.pipes.get_album_costs import \
    GetAlbumCosts
from track_analysis.components.track_analysis.features.data_generation.pipes.get_audio_files import \
    GetAudioFiles
from track_analysis.components.track_analysis.features.data_generation.pipes.get_invalid_cache import \
    HandleInvalidCache
from track_analysis.components.track_analysis.features.data_generation.pipes.handle_rows_with_missing_data import \
    FillMissingHeadersPipe
from track_analysis.components.track_analysis.features.data_generation.pipes.load_cache import LoadCache
from track_analysis.components.track_analysis.features.data_generation.pipes.load_energy_calculator import \
    LoadEnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.pipes.load_processors import \
    CreateProcessors
from track_analysis.components.track_analysis.features.data_generation.pipes.make_csv import MakeCSV
from track_analysis.components.track_analysis.features.data_generation.pipes.preprocess_data import \
    PreprocessData
from track_analysis.components.track_analysis.features.data_generation.pipes.redo_headers import RedoHeaders
from track_analysis.components.track_analysis.features.data_generation.pipes.remove_invalid_cached_entries import \
    RemoveInvalidCachedEntries
from track_analysis.components.track_analysis.features.data_generation.helpers.key_extractor import KeyExtractor
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor


@dataclasses.dataclass(frozen=True)
class PipelineConfiguration:
    num_workers: int
    hop_length: int
    n_fft: int


class BuildLibraryDataCSVPipeline(AbPipeline):
    def __init__(self,
                 logger: HoornLogger,
                 filehandler: FileHandler,
                 tag_extractor: TagExtractor,
                 string_utils: StringUtils,
                 key_extractor: KeyExtractor,
                 max_rate_cache: MaxRateCache,
                 configuration: PipelineConfiguration):
        self._logger = logger
        self._filehandler = filehandler
        self._key_extractor = key_extractor
        self._string_utils = string_utils
        self._num_workers: int = configuration.num_workers

        self._metadata_provider: MetadataDFBuilder = MetadataDFBuilder(tag_extractor)
        self._results_mapper: ResultsMapper = ResultsMapper(FEATURE_TO_HEADER_MAPPING)
        self._key_data_builder: KeyDataFramesBuilder = KeyDataFramesBuilder()

        self._hop_length = configuration.hop_length
        self._n_fft = configuration.n_fft
        self._max_rate_cache = max_rate_cache

        super().__init__(logger)

    def build_pipeline(self):
        def __exit_if_no_files_to_process(context: LibraryDataGenerationPipelineContext) -> bool:
            return (
                    len(context.filtered_audio_file_paths) <= 0
                    and len(context.invalid_cached_paths) <= 0
                    and len(context.missing_headers.keys()) <= 0
                    and len(context.headers_to_refill) <= 0
            )

        def __exit_if_at_energy_loading(context: LibraryDataGenerationPipelineContext) -> bool:
            return context.end_at_energy_calculation_loading

        self._add_step(GetAlbumCosts())
        self._add_step(LoadCache(self._logger))
        self._add_step(LoadEnergyCalculator(self._logger))
        self._add_exit_check(__exit_if_at_energy_loading)
        self._add_step(CreateProcessors(
            logger=self._logger,
            num_workers=self._num_workers,
            hop_length=self._hop_length,
            n_fft=self._n_fft,
            max_rate_cache=self._max_rate_cache,
            key_extractor=self._key_extractor,
        ))
        self._add_step(FilterCache(self._logger))
        self._add_step(GetAudioFiles(self._logger, self._filehandler))
        self._add_step(FilterFiles(self._logger))
        self._add_step(HandleInvalidCache(self._logger))
        self._add_exit_check(__exit_if_no_files_to_process)
        self._add_step(BatchProcessNewTracks(
            logger=self._logger,
            key_data_builder=self._key_data_builder,
            results_mapper=self._results_mapper,
            metadata_builder=self._metadata_provider,
        ))
        self._add_step(RemoveInvalidCachedEntries(self._logger))
        self._add_step(FillMissingHeadersPipe(self._logger))
        self._add_step(RedoHeaders(self._logger))
        self._add_step(PreprocessData(self._logger, self._string_utils))
        self._add_step(MakeCSV(self._logger))
