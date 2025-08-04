import traceback
from typing import Callable, Any

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

_SEPARATOR = "LogUtils"

def log_and_handle_exception(logger: HoornLogger, message: str, action: Callable, *args: Any, **kwargs: Any) -> Any:
    """Helper to wrap an action in a try-except block for consistent error logging."""
    try:
        return action(*args, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"{message}: {e}\n{tb}", separator=_SEPARATOR)
        return None
