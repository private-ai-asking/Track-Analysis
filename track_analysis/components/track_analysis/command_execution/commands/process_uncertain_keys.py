from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.scrobble_linking.processor.uncertain_keys_processor import \
    UncertainKeysProcessor


class ProcessUncertainKeysCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, uncertain_keys_processor: UncertainKeysProcessor):
        super().__init__(logger)
        self._uncertain_keys_processor = uncertain_keys_processor

    @property
    def default_command_keys(self) -> List[str]:
        return ["process_uncertain_keys", "puk"]

    @property
    def command_description(self) -> str:
        return "Processes the scrobble linking service's uncertain UUIDs interactively."

    def execute(self, arguments: T) -> P:
        self._uncertain_keys_processor.process()
