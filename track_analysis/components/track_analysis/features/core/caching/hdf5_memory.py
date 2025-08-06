import inspect
from functools import wraps
from pathlib import Path
from typing import Union, List, Callable

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.caching.cache_manager import HDF5CacheManager


class HDF5Memory:
    """
    A class that mimics the joblib.Memory API to provide a caching decorator
    backed by a thread-safe HDF5CacheManager.
    """
    def __init__(self, cache_path: Union[str, Path]):
        """Initializes the memory object and its underlying cache manager."""
        self._cache_manager = HDF5CacheManager(cache_path)

    def set_logger(self, logger: HoornLogger):
        self._cache_manager.set_logger(logger)

    def log_stats(self, top_n: int = 5):
        self._cache_manager.log_statistics(top_n=top_n)

    def cache(self, identifier_arg: str, ignore: List[str] = None) -> Callable:
        """
        A decorator factory that caches the output of a function in the HDF5 file.
        """
        ignored_args = set(ignore) if ignore else set()

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                all_args = bound_args.arguments

                if identifier_arg not in all_args:
                    raise ValueError(f"Identifier argument '{identifier_arg}' not found in call to {func.__name__}")

                identifier_value = str(all_args[identifier_arg])
                feature_name = func.__name__

                params_for_key = {
                    key: value
                    for key, value in all_args.items()
                    if key not in ignored_args and key != identifier_arg
                }

                # --- Step 2: A single, atomic "get-or-compute" block ---
                cached_result = self._cache_manager.get(identifier_value, feature_name, params_for_key)

                if cached_result is not None:
                    return cached_result
                else:
                    result = func(*args, **kwargs)
                    self._cache_manager.put(identifier_value, feature_name, params_for_key, result)

                    return result
            return wrapper
        return decorator
