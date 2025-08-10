from typing import List

from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_manager import CommandManager
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.command_execution.command_factory import CommandFactory
from track_analysis.components.track_analysis.command_execution.commands.execute_with_profiler import \
    ExecuteWithProfilerCommand
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class CommandLineInitiator:
    def __init__(self,
                 logger: HoornLogger,
                 cli: CommandLineInterface,
                 command_factory: CommandFactory,
                 configuration: TrackAnalysisConfigurationModel):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._configuration = configuration
        self._cli = cli
        self._commands = command_factory.get_commands()
        self._command_manager: CommandManager = CommandManager(self._logger)
        self._profile_command = ExecuteWithProfilerCommand(self._logger, configuration.paths.benchmark_directory)

    def _execute_command_normal(self, command: CommandExecutionModel) -> None:
        self._command_manager.execute_command(command, None)

    def _execute_command_profile(self, command: CommandExecutionModel) -> None:
        self._command_manager.execute_command(self._profile_command, arguments=command)

    def initialize(self) -> None:
        for command in self._commands:
            keys: List[str] = command.default_command_keys
            description: str = command.command_description

            profile_keys: List[str] = [
                f"{key}-p"
                for key in keys
            ]
            profile_description = f"[ALSO PROFILES] {description}"

            self._cli.add_command(keys, description, self._execute_command_normal, arguments=[command])
            self._cli.add_command(profile_keys, profile_description, self._execute_command_profile, arguments=[command])
