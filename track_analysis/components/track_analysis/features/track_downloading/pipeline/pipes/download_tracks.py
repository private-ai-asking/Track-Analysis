from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.track_downloading.api.music_download_interface import \
    MusicDownloadInterface
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline_context import \
    DownloadPipelineContext


class DownloadTracks(IPipe):
    """Pipe to gather all the tracks from the csv."""
    def __init__(self, logger: HoornLogger, downloader: MusicDownloadInterface):
        self._logger = logger
        self._separator: str = "DownloadPipeline.DownloadTracks"

        self._downloader: MusicDownloadInterface = downloader

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, data: DownloadPipelineContext) -> DownloadPipelineContext:
        data.downloaded_tracks = self._downloader.download_tracks(data.tracks_to_download)

        return data
