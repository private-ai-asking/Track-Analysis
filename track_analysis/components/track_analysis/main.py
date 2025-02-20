from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.logging import HoornLogger, LogType

if __name__ == "__main__":
    logger: HoornLogger = HoornLogger(
        min_level=LogType.DEBUG,
        separator_root="TrackAnalysis",
        max_separator_length=30)

    cmd: CommandLineInterface = CommandLineInterface(logger)
    cmd.start_listen_loop()
