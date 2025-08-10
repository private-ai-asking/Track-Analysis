from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService


class TestParametersCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, scrobble_linker: ScrobbleLinkerService):
        super().__init__(logger)
        self._scrobble_linker = scrobble_linker

    @property
    def default_command_keys(self) -> List[str]:
        return ["test_parameters", "tp"]

    @property
    def command_description(self) -> str:
        return "Finds out the best parameters for the scrobble linking service's algorithm."

    def execute(self, arguments: T) -> P:
        self._scrobble_linker.test_parameters()
