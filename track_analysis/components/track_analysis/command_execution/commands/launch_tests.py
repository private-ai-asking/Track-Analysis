from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import T, \
    P
from track_analysis.components.md_common_python.py_common.testing import TestCoordinator
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel


class LaunchTestsCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, test_coordinator: TestCoordinator):
        super().__init__(logger)
        self._test_coordinator = test_coordinator

    @property
    def default_command_keys(self) -> List[str]:
        return ["launch_tests", "lt"]

    @property
    def command_description(self) -> str:
        return "Launches the test framework CLI."

    def execute(self, arguments: T) -> P:
        self._test_coordinator.start_test_cli()
