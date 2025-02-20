from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.app import App
from track_analysis.components.track_analysis.initialize_logger import get_logger

LOGGER: HoornLogger = get_logger()

if __name__ == "__main__":
    app: App = App(LOGGER)
    app.run()
