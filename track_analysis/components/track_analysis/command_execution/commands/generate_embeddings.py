from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService


class GenerateEmbeddingsCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, scrobble_linker: ScrobbleLinkerService):
        super().__init__(logger)
        self._scrobble_linker = scrobble_linker

    @property
    def default_command_keys(self) -> List[str]:
        return ["generate_embeddings", "ge"]

    @property
    def command_description(self) -> str:
        return "Generates embeddings for the library to be used when similarity matching scrobbles."

    def execute(self, arguments: T) -> P:
        self._scrobble_linker.build_embeddings_for_library()
