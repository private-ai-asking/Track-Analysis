from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger, LogType, FileHoornLogOutput, \
    DefaultHoornLogOutput


def _get_user_local_appdata_dir() -> Path:
    # Get OS Name
    import platform
    system = platform.system()

    if system == "Windows":
        return Path(Path.home()).joinpath("AppData", "Local")
    elif system == "Darwin":
        return Path(Path.home()).joinpath("Library", "Application Support")
    elif system == "Linux":
        return Path(Path.home()).joinpath(".local", "share")

    raise ValueError("Unsupported operating system")

def get_logger() -> HoornLogger:
    max_sep_length = 40

    file_output: FileHoornLogOutput = FileHoornLogOutput(
        log_directory=_get_user_local_appdata_dir().joinpath("Track Analysis", "logs"),
        max_logs_to_keep=10,
        max_separator_length=max_sep_length
    )

    return HoornLogger(
        outputs=[file_output, DefaultHoornLogOutput(max_separator_length=max_sep_length)],
        min_level=LogType.TRACE,
        separator_root="TrackAnalysis",
        max_separator_length=max_sep_length)
