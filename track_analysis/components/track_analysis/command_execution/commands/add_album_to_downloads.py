from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.track_downloading.api.metadata_api import MetadataAPI


class AddAlbumToDownloadsCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, metadata_api: MetadataAPI):
        super().__init__(logger)
        self._metadata_api = metadata_api

    @property
    def default_command_keys(self) -> List[str]:
        return ["add_album_to_downloads", "aatd"]

    @property
    def command_description(self) -> str:
        return "Adds an album to the downloads.csv file."

    def execute(self, arguments: T) -> P:
        self._metadata_api.add_album_to_downloads()
