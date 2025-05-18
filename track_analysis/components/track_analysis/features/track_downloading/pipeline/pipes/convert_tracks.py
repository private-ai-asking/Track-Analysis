import os
import threading
import traceback
import asyncio
from pathlib import Path

from track_analysis.components.md_common_python.py_common.command_handling import CommandHelper
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.constants import FFMPEG_PATH
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline_context import (
    DownloadPipelineContext,
)


class ConvertTracks(IPipe):
    """Pipe to gather all the tracks from the csv."""
    def __init__(self, logger: HoornLogger, command_helper: CommandHelper):
        self._logger = logger
        self._separator: str = "DownloadPipeline.ConvertTracks"
        self._command_helper: CommandHelper = command_helper
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, data: DownloadPipelineContext) -> DownloadPipelineContext:
        for track in data.downloaded_tracks:
            try:
                original_path: Path = track.path
                final_path:    Path = original_path.with_suffix('.flac')
                temp_path:     Path = final_path.with_suffix('.flac.tmp')

                cmd = [
                    "-y",
                    "-i", f"\"{str(original_path)}\"",
                    "-c:a", "flac",
                    "-f", "flac",
                    f"\"{str(temp_path)}\""
                ]

                # run the async FFmpeg call on its own thread
                def _run_ffmpeg():
                    asyncio.run(
                        self._command_helper.execute_command_v2_async(
                            FFMPEG_PATH,
                            cmd,
                            hide_console=True,
                            keep_open=False
                        )
                    )

                thread = threading.Thread(target=_run_ffmpeg)
                thread.start()
                thread.join()

                self._logger.info(
                    f"Converted {original_path.name} → {final_path.name}",
                    separator=self._separator
                )

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
