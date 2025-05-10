from pathlib import Path

ROOT_MUSIC_LIBRARY: Path = Path("W:\media\music\[02] organized") # NORMAL
# ROOT_MUSIC_LIBRARY: Path = Path("W:\\media\\music\\[03] test") # TESt
OUTPUT_DIRECTORY: Path = Path("X:\Track Analysis\output")

DATA_DIRECTORY: Path = Path("X:\Track Analysis\data")
CACHE_DIRECTORY: Path = Path("X:\Track Analysis\cache")
BENCHMARK_DIRECTORY: Path = Path(r"X:\Track Analysis\track_analysis\benchmarks")

DEBUG: bool = False
VERBOSE: bool = False

MINIMUM_CONFIDENCE_THRESHOLD: float = 85  # 0-100

CLEAR_CACHE: bool = True  # For developing purposes, clears the cache each time to reset -- for profiling.
DELETE_FINAL_DATA_BEFORE_START: bool = True  # For developing purposes; removes `enriched_scrobbles.csv` on start.
TEST_SAMPLE_SIZE: int = 1500

NO_MATCH_LABEL: str = "<NO MATCH>"
