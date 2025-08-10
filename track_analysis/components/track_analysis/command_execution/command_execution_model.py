import abc
from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T


class CommandExecutionModel(CommandInterface):
    def __init__(self, logger: HoornLogger):
        super().__init__(logger)

    @property
    @abc.abstractmethod
    def default_command_keys(self) -> List[str]:
        """Returns the default command keys for this command."""

    @property
    @abc.abstractmethod
    def command_description(self) -> str:
        """Returns the description of this command."""

    @abc.abstractmethod
    def execute(self, arguments: None) -> None:
        pass

    def unexecute(self, arguments: T) -> None:
        raise NotImplementedError()
