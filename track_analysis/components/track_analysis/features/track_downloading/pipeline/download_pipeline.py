from pathlib import Path

from track_analysis.components.md_common_python.py_common.command_handling import CommandHelper
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.track_analysis.features.track_downloading.api.music_download_interface import \
    MusicDownloadInterface
from track_analysis.components.track_analysis.features.track_downloading.pipeline.pipes.convert_tracks import \
    ConvertTracks
from track_analysis.components.track_analysis.features.track_downloading.pipeline.pipes.download_tracks import \
    DownloadTracks
from track_analysis.components.track_analysis.features.track_downloading.pipeline.pipes.gather_tracks import \
    GatherTracks


class DownloadPipeline(AbPipeline):
    """Pipeline for the downloading of YT Music Tracks."""
    def __init__(self, logger: HoornLogger, downloader: MusicDownloadInterface, command_helper: CommandHelper, ffmpeg_path: Path):
        self._downloader = downloader
        self._command_helper = command_helper
        self._ffmpeg_path = ffmpeg_path

        super().__init__(logger, pipeline_descriptor="DownloadingPipeline")

    def build_pipeline(self):
        self._add_step(GatherTracks(self._logger))
        self._add_step(DownloadTracks(self._logger, self._downloader))
        self._add_step(ConvertTracks(self._logger, self._command_helper, self._ffmpeg_path))
