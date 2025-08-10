import os
import subprocess
import traceback
from pathlib import Path

from track_analysis.components.md_common_python.py_common.command_handling import CommandHelper
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline_context import (
    DownloadPipelineContext,
)


class ConvertTracks(IPipe):
    """Pipe to gather all the tracks from the csv."""
    def __init__(self, logger: HoornLogger, command_helper: CommandHelper, ffmpeg_path: Path):
        self._logger = logger
        self._separator: str = "DownloadPipeline.ConvertTracks"
        self._command_helper: CommandHelper = command_helper
        self._ffmpeg_path: Path = ffmpeg_path
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, data: DownloadPipelineContext) -> DownloadPipelineContext:
        for track in data.downloaded_tracks:
            try:
                original_path: Path = track.path
                final_path:    Path = original_path.with_suffix('.flac')
                temp_path:     Path = final_path.with_suffix('.flac.tmp')

                cmd = [
                    "-y",
                    "-i", str(original_path),
                    "-c:a", "flac",
                    "-f", "flac",
                    str(temp_path)
                ]

                # synchronous, never hangs
                subprocess.run(
                    [str(self._ffmpeg_path)] + cmd,
                    check=True
                )

                self._logger.info(f"Converted {original_path.name} → {final_path.name}", ...)

                # now swap files
                os.remove(original_path)
                temp_path.rename(final_path)
                track.path = final_path

            except Exception:
                tb = traceback.format_exc()
                self._logger.error(
                    f"Error converting {track.path.name}:\n{tb}",
                    separator=self._separator
                )
                continue

        return data
