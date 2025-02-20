from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.initialize_logger import get_logger

if __name__ == "__main__":
    logger: HoornLogger = get_logger()

    cmd: CommandLineInterface = CommandLineInterface(logger)
    cmd.start_listen_loop()
