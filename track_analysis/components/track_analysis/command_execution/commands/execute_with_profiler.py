from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.library.viztrace_profiler import run_with_profiling


class ExecuteWithProfilerCommand(CommandInterface):
    def __init__(self, logger: HoornLogger, benchmark_dir: Path):
        super().__init__(logger)
        self._benchmark_dir = benchmark_dir

    def execute(self, command: CommandExecutionModel) -> P:
        command_name = command.__class__.__name__
        run_with_profiling(lambda: command.execute(None), category=command_name, benchmark_dir=self._benchmark_dir)

    def unexecute(self, arguments: T) -> None:
        raise NotImplementedError()
