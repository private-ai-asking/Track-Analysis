from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.pipeline.pipes.get_audio_files import GetAudioFiles
from track_analysis.components.track_analysis.pipeline.pipes.get_metadata import GetAudioMetadata
from track_analysis.components.track_analysis.pipeline.pipes.load_cache import LoadCache
from track_analysis.components.track_analysis.pipeline.pipes.make_csv import MakeCSV
from track_analysis.components.track_analysis.pipeline.pipes.match_paths import MatchPaths
from track_analysis.components.track_analysis.pipeline.pipes.preprocess_data import PreprocessData


class LocatePathsPipeline(AbPipeline):
    def __init__(self, logger: HoornLogger, filehandler: FileHandler, tag_extractor: TagExtractor):
        self._logger = logger
        self._filehandler = filehandler
        self._tag_extractor = tag_extractor
        super().__init__()

    def build_pipeline(self):
        self._add_step(LoadCache(self._logger))
        self._add_step(GetAudioFiles(self._logger, self._filehandler))
        self._add_step(GetAudioMetadata(self._logger, self._tag_extractor))
        self._add_step(PreprocessData(self._logger))
        self._add_step(MatchPaths(self._logger))
        self._add_step(MakeCSV(self._logger))
