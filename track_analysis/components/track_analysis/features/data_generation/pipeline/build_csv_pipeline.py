from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.add_advanced_metadata import AddAdvancedMetadata
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.get_invalid_cache import GetInvalidCache
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.filter_files import FilterFiles
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.get_album_costs import GetAlbumCosts
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.get_audio_files import GetAudioFiles
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.get_metadata import GetAndBuildAudioMetadata
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.load_cache import LoadCache
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.make_csv import MakeCSV
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipes.preprocess_data import PreprocessData


class BuildLibraryDataCSVPipeline(AbPipeline):
    def __init__(self,
                 logger: HoornLogger,
                 filehandler: FileHandler,
                 tag_extractor: TagExtractor,
                 audio_file_handler: AudioFileHandler,
                 audio_calculator: AudioCalculator,
                 string_utils: StringUtils):
        self._logger = logger
        self._filehandler = filehandler
        self._tag_extractor = tag_extractor
        self._audio_file_handler = audio_file_handler
        self._audio_calculator = audio_calculator
        self._string_utils = string_utils
        super().__init__(logger)

    def build_pipeline(self):
        def __exit_if_no_files_to_process(context: LibraryDataGenerationPipelineContext) -> bool:
            return len(context.filtered_audio_file_paths) <= 0 and len(context.invalid_cached_paths) <= 0

        self._add_step(GetAlbumCosts())
        self._add_step(LoadCache(self._logger))
        self._add_step(GetAudioFiles(self._logger, self._filehandler))
        self._add_step(FilterFiles(self._logger))
        self._add_step(GetInvalidCache(self._logger))
        self._add_exit_check(__exit_if_no_files_to_process)
        self._add_step(GetAndBuildAudioMetadata(self._logger, self._tag_extractor))
        self._add_step(AddAdvancedMetadata(self._logger, self._audio_file_handler, self._audio_calculator))
        self._add_step(PreprocessData(self._logger, self._string_utils))
        self._add_step(MakeCSV(self._logger))
