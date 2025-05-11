from pathlib import Path

ROOT_MUSIC_LIBRARY: Path = Path("W:\media\music\[02] organized") # NORMAL
# ROOT_MUSIC_LIBRARY: Path = Path("W:\\media\\music\\[03] test") # TESt
OUTPUT_DIRECTORY: Path = Path("X:\Track Analysis\output")

DATA_DIRECTORY: Path = Path("X:\Track Analysis\data")
CACHE_DIRECTORY: Path = Path("X:\Track Analysis\cache")
BENCHMARK_DIRECTORY: Path = Path(r"X:\Track Analysis\track_analysis\benchmarks")

DEBUG: bool = True
VERBOSE: bool = False

MINIMUM_CONFIDENCE_THRESHOLD: float = 90  # 0-100

CLEAR_CACHE: bool = True  # For developing purposes, clears the cache each time to reset -- for profiling.
DELETE_FINAL_DATA_BEFORE_START: bool = True  # For developing purposes; removes `enriched_scrobbles.csv` on start.
TEST_CACHE_BUILDER_MODE: bool = True  # For developing purposes; auto accepts every scrobble match to see data in cache clearly.

TEST_SAMPLE_SIZE: int = 5000  # The number of scrobbles to load (only applies to scrobbles, the library itself will always be fully loaded)

NO_MATCH_LABEL: str = "<NO MATCH>"
