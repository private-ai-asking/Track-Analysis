import time

from track_analysis.components.md_common_python.py_common.logging import HoornLogger, DefaultHoornLogOutput, LogType
from track_analysis.components.md_common_python.py_common.time_handling.time_utils import time_operation, TimeUtils

max_sep_length = 65

logger = HoornLogger(
    outputs=[DefaultHoornLogOutput(max_separator_length=max_sep_length)],
    min_level=LogType.TRACE,
    separator_root="TrackAnalysis",
    max_separator_length=max_sep_length)


@time_operation(logger=logger, time_utils=TimeUtils(), separator="Test")
def _sleep_func(n: int):
    time.sleep(n)

if __name__ == "__main__":
    _sleep_func(3)
    _sleep_func(5)
