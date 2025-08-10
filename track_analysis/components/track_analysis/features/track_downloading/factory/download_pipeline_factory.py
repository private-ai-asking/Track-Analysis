from track_analysis.components.md_common_python.py_common.command_handling import CommandHelper
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.track_downloading.api.music_download_interface import \
    MusicDownloadInterface
from track_analysis.components.track_analysis.features.track_downloading.api.ytdlp_music_downloader import \
    YTDLPMusicDownloader
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline import \
    DownloadPipeline
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class DownloadPipelineFactory:
    """Creates the download pipeline."""
    def __init__(self,
                 logger: HoornLogger,
                 command_helper: CommandHelper,
                 configuration: TrackAnalysisConfigurationModel):
        self._logger = logger
        self._command_helper = command_helper
        self._configuration = configuration

    def create(self) -> DownloadPipeline:
        downloader: MusicDownloadInterface = YTDLPMusicDownloader(self._logger, self._configuration.paths.music_track_downloads_dir, self._configuration.paths.cookies_file, self._configuration.paths.ffmpeg_path)
        return DownloadPipeline(
            self._logger,
            downloader,
            self._command_helper,
            self._configuration.paths.ffmpeg_path,
        )
