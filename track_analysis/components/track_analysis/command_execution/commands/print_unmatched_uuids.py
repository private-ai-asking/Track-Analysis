from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.scrobble_linking.get_unmatched_library_tracks import \
    UnmatchedLibraryTracker


class PrintUnmatchedUUIDsCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, unmatched_util: UnmatchedLibraryTracker):
        super().__init__(logger)
        self._unmatched_util = unmatched_util

    @property
    def default_command_keys(self) -> List[str]:
        return ["print_unmatched_uuids", "puu"]

    @property
    def command_description(self) -> str:
        return "Logs library tracks without associated scrobbles."

    def execute(self, arguments: T) -> P:
        self._unmatched_util.print_unmatched_tracks()
