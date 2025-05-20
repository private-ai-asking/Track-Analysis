from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.constants import DEBUG, VERBOSE
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import LibraryDataGenerationPipelineContext


class GetAudioFiles(IPipe):
    def __init__(self, logger: HoornLogger, file_handler: FileHandler):
        self._separator = "BuildCSV.GetAudioFilesPipe"

        self._logger = logger
        self._file_handler = file_handler
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Getting audio file paths.", separator=self._separator)

        track_paths = self._file_handler.get_children_paths_fast(data.source_dir, [".flac", ".mp3", ".opus"], recursive=True)

        # TODO - Remove 20 limit after optimization
        data.all_audio_file_paths = track_paths

        self._logger.trace("Successfully gotten audio file paths.", separator=self._separator)

        if VERBOSE:
            self._logger.debug(f"Audio files: {track_paths}", separator=self._separator)
        return data
