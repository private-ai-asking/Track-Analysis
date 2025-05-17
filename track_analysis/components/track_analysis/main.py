import os

from track_analysis.components.md_common_python.py_common.logging import HoornLogger, LogType, HoornLoggerBuilder
from track_analysis.components.track_analysis.app import App
from track_analysis.components.track_analysis.constants import DEBUG, VERBOSE, CPU_COUNT

os.environ["LOKY_MAX_CPU_COUNT"] = str(CPU_COUNT)

if __name__ == "__main__":
    min_level: LogType = LogType.INFO

    if DEBUG:
        min_level = LogType.DEBUG

        if VERBOSE:
            min_level = LogType.TRACE

    builder = HoornLoggerBuilder("TrackAnalysis", max_separator_length=65)
    (builder
     .build_file_based_output()
     .build_gui_output(base_batch_size=1500, max_batch_size=8000))

    logger: HoornLogger = builder.get_logger(min_level)

    app: App = App(logger)
    app.run()
