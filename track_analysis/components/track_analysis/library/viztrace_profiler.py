from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from viztracer import VizTracer

T = TypeVar("T")


def run_with_profiling(func: Callable[[], T], category: str, benchmark_dir: Path) -> T:
    """
    Runs the given function under VizTracer profiling and writes a benchmark file.

    :param func: A no-argument function to execute and profile.
    :param category: The profiling category (used to name the subdirectory).
    :return: The return value of the function.
    """
    # Prepare timestamp and paths
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d-%H_%M_%S")
    benchmark_dir = benchmark_dir.joinpath(category)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_dir.joinpath(f"{timestamp}-benchmark.json")

    # Run with VizTracer
    with VizTracer(
            output_file=str(benchmark_path),
            min_duration=1,
            ignore_c_function=True,
            ignore_frozen=True
    ) as _:
        result = func()

    return result
