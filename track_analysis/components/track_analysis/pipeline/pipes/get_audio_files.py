from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.constants import DEBUG
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class GetAudioFiles(IPipe):
    def __init__(self, logger: HoornLogger, file_handler: FileHandler):
        self._separator = "GetAudioFilesPipe"

        self._logger = logger
        self._file_handler = file_handler
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Getting audio file paths.", separator=self._separator)

        track_paths = self._file_handler.get_children_paths_fast(data.source_dir, [".flac", ".mp3", ".opus"], recursive=True)

        if DEBUG:
            data.audio_file_paths = track_paths[:10]
        else: data.audio_file_paths = track_paths

        self._logger.trace("Successfully gotten audio file paths.", separator=self._separator)
        self._logger.debug(f"Audio files: {track_paths}", separator=self._separator)
        return data
