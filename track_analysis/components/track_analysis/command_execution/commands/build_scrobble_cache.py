from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService


class BuildScrobbleCacheCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, scrobble_linker: ScrobbleLinkerService):
        super().__init__(logger)
        self._scrobble_linker = scrobble_linker

    @property
    def default_command_keys(self) -> List[str]:
        return ["build_scrobble_cache", "bsc"]

    @property
    def command_description(self) -> str:
        return "Builds the scrobble cache for use when linking scrobbles to the library data."

    def execute(self, arguments: T) -> P:
        self._scrobble_linker.build_cache()
