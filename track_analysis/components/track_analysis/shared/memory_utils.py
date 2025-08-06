import ctypes
import gc

import numpy as np
import psutil

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared.log_utils import log_and_handle_exception

_SEPARATOR = "MemoryUtils"

def trim_and_defrag(logger: HoornLogger, after_description: str = ""):
    gc.collect()
    log_and_handle_exception(
        logger,
        "Failed to call SetProcessWorkingSetSize",
        lambda: ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    )
    try:
        scratch = np.empty((52_000_000,), dtype=np.float32)
        del scratch
        gc.collect()
    except Exception:
        pass

    vm = psutil.virtual_memory()
    proc = psutil.Process()
    description = f"[{after_description}] " if after_description else ""
    logger.debug(f"{description}Total RAM:   {vm.total >> 20} MiB", separator=_SEPARATOR)
    logger.debug(f"{description}Available:   {vm.available >> 20} MiB", separator=_SEPARATOR)
    logger.debug(f"{description}Process RSS: {proc.memory_info().rss >> 20} MiB", separator=_SEPARATOR)
