from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.batch_process_new_tracks import \
    BatchProcessNewTracks
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.filter_cache import FilterCache
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.filter_files import FilterFiles
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.get_album_costs import \
    GetAlbumCosts
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.get_audio_files import \
    GetAudioFiles
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.get_invalid_cache import \
    GetInvalidCache
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.handle_rows_with_missing_data import \
    HandleRowsWithMissingData
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.load_cache import LoadCache
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.load_energy_calculator import \
    LoadEnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.make_csv import MakeCSV
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.preprocess_data import \
    PreprocessData
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.redo_headers import RedoHeaders
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.remove_invalid_cached_entries import \
    RemoveInvalidCachedEntries
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor


class BuildLibraryDataCSVPipeline(AbPipeline):
    def __init__(self,
                 logger: HoornLogger,
                 filehandler: FileHandler,
                 tag_extractor: TagExtractor,
                 audio_file_handler: AudioFileHandler,
                 audio_calculator: AudioCalculator,
                 string_utils: StringUtils,
                 num_workers: int,
                 num_workers_refill: int):
        self._logger = logger
        self._filehandler = filehandler
        self._tag_extractor = tag_extractor
        self._audio_file_handler = audio_file_handler
        self._audio_calculator = audio_calculator
        self._string_utils = string_utils
        self._num_workers: int = num_workers
        self._num_workers_refill: int = num_workers_refill
        super().__init__(logger)

    def build_pipeline(self):
        def __exit_if_no_files_to_process(context: LibraryDataGenerationPipelineContext) -> bool:
            return (
                    len(context.filtered_audio_file_paths) <= 0
                    and len(context.invalid_cached_paths) <= 0
                    and len(context.missing_headers.keys()) <= 0
                    and len(context.refill_headers.keys()) <= 0
            )

        self._add_step(GetAlbumCosts())
        self._add_step(LoadCache(self._logger))
        self._add_step(LoadEnergyCalculator(self._logger))
        self._add_step(FilterCache(self._logger))
        self._add_step(GetAudioFiles(self._logger, self._filehandler))
        self._add_step(FilterFiles(self._logger))
        self._add_step(GetInvalidCache(self._logger))
        self._add_exit_check(__exit_if_no_files_to_process)
        self._add_step(BatchProcessNewTracks(
            self._logger,
            self._audio_file_handler,
            self._tag_extractor,
            self._audio_calculator,
        ))
        self._add_step(RemoveInvalidCachedEntries(self._logger))
        self._add_step(HandleRowsWithMissingData(self._logger, self._audio_file_handler))
        self._add_step(RedoHeaders(self._logger, self._audio_file_handler, self._num_workers_refill))
        self._add_step(PreprocessData(self._logger, self._string_utils))
        self._add_step(MakeCSV(self._logger))
