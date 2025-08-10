import dataclasses
import inspect
import time
from functools import wraps
from pathlib import Path
from typing import Union, List, Callable, Set, Any, Dict, Generic, TypeVar, ParamSpec

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared.caching.cache_manager import HDF5CacheManager

T = TypeVar("T")
P = ParamSpec("P")


@dataclasses.dataclass(frozen=True)
class TimedCacheResult(Generic[T]):
    value: T
    retrieved_from_cache: bool
    time_waiting: float
    time_processing: float


class HDF5Memory:
    """
    A class that mimics the joblib.Memory API to provide a caching decorator
    backed by a thread-safe HDF5CacheManager.
    """
    def __init__(self):
        self._cache_manager: HDF5CacheManager | None = None

    def close(self) -> None:
        self._cache_manager.close()

    def set_logger(self, logger: HoornLogger):
        self._cache_manager.set_logger(logger)

    def instantiate(self, cache_path: Union[str, Path]):
        self._cache_manager = HDF5CacheManager(cache_path)

    def log_stats(self, top_n: int = 5):
        self._cache_manager.log_statistics(top_n=top_n)

    def cache(self, identifier_arg: str, ignore: List[str] = None) -> Callable:
        """
        A decorator factory that caches the output of a function in the HDF5 file.
        """
        ignored_args_set = set(ignore) if ignore else set()

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                if self._cache_manager is None:
                    raise RuntimeError("Trying to cache without instantiated cache manager.")

                cache_keys = self._get_cache_keys(
                    func, identifier_arg, ignored_args_set, *args, **kwargs
                )
                cached_result = self._get_cached_result(cache_keys)

                if cached_result is not None:
                    return cached_result

                result = func(*args, **kwargs)
                self._put_result_in_cache(cache_keys, result)
                return result
            return wrapper
        return decorator

    def timed_cache(self, identifier_arg: str, ignore: list[str] | None = None) -> Callable[[Callable[P, T]], Callable[P, TimedCacheResult[T]]]:
        """
        A decorator factory that caches the output of a function in the HDF5 file and provides time information.
        """
        ignored_args_set = set(ignore) if ignore else set()

        def decorator(func: Callable[P, T]) -> Callable[P, TimedCacheResult[T]]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> TimedCacheResult[T]:
                if self._cache_manager is None:
                    raise RuntimeError("Trying to cache without instantiated cache manager.")

                time_start = time.perf_counter()

                cache_keys = self._get_cache_keys(
                    func, identifier_arg, ignored_args_set, *args, **kwargs
                )
                processing_duration = time.perf_counter() - time_start

                io_start = time.perf_counter()
                cached_result = self._get_cached_result(cache_keys)
                io_duration = time.perf_counter() - io_start

                if cached_result is not None:
                    return TimedCacheResult(
                        value=cached_result,
                        retrieved_from_cache=True,
                        time_waiting=io_duration,
                        time_processing=processing_duration,
                    )

                processing_start = time.perf_counter()
                result = func(*args, **kwargs)
                processing_duration = processing_duration + (time.perf_counter() - processing_start)

                io_start = time.perf_counter()
                self._put_result_in_cache(cache_keys, result)
                io_duration = io_duration + (time.perf_counter() - io_start)
                return TimedCacheResult(
                    value=result,
                    retrieved_from_cache=False,
                    time_waiting=io_duration,
                    time_processing=processing_duration,
                )
            return wrapper
        return decorator

    @staticmethod
    def _get_cache_keys(
            func: Callable,
            identifier_arg: str,
            ignored_args: Set[str],
            *args,
            **kwargs
    ) -> Dict[str, Any]:
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

        cache_keys = {
            'identifier': identifier_value,
            'feature': feature_name,
            'params': params_for_key,
        }

        return cache_keys

    def _get_cached_result(self, cache_keys: Dict[str, Any]) -> Any | None:
        cached_result = self._cache_manager.get(
            cache_keys['identifier'], cache_keys['feature'], cache_keys['params']
        )
        return cached_result

    def _put_result_in_cache(self, cache_keys: Dict[str, Any], result: Any) -> None:
        self._cache_manager.put(
            cache_keys['identifier'],
            cache_keys['feature'],
            cache_keys['params'],
            result
        )
