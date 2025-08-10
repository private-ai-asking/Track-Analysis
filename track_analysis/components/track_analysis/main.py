import os
import signal
import warnings

from track_analysis.components.track_analysis.constants import PHYSICAL_CPU_COUNT, CONFIGURATION_PATH
from track_analysis.components.track_analysis.library.configuration.configuration_loader import ConfigurationLoader

os.environ['LOKY_MAX_CPU_COUNT'] = str(PHYSICAL_CPU_COUNT)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r".*threadpoolctl.*"
)

from track_analysis.components.md_common_python.py_common.logging import HoornLogger, LogType, HoornLoggerBuilder
from track_analysis.components.track_analysis.app import App


if __name__ == "__main__":
    configuration_loader: ConfigurationLoader = ConfigurationLoader(CONFIGURATION_PATH)
    configuration = configuration_loader.get_configuration()

    min_level: LogType = LogType.INFO

    if configuration.development.debug:
        min_level = LogType.DEBUG

        if configuration.development.verbose:
            min_level = LogType.TRACE

    builder = HoornLoggerBuilder("TrackAnalysis", max_separator_length=65)
    (builder
     .build_file_based_output(max_logs_to_keep=30, buffer_limit=2500)
     .build_gui_output())

    logger: HoornLogger = builder.get_logger(min_level)

    app: App = App(logger, configuration)
    app.run()

    signal.signal(signal.SIGINT, app.on_exit)
    signal.signal(signal.SIGTERM, app.on_exit)
