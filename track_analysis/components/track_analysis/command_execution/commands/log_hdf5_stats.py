from typing import List

from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.shared_objects import MEMORY


class LogHDF5StatsCommand(CommandExecutionModel):
    @property
    def default_command_keys(self) -> List[str]:
        return ["log_hdf5_stats", "lh5"]

    @property
    def command_description(self) -> str:
        return "Logs interesting statistics about the processing cache."

    def execute(self, arguments: T) -> P:
        MEMORY.log_stats()
