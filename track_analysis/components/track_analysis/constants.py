from pathlib import Path

ROOT_MUSIC_LIBRARY: Path = Path("W:\media\music\[02] organized") # NORMAL
# ROOT_MUSIC_LIBRARY: Path = Path("W:\\media\\music\\[03] test") # TESt
OUTPUT_DIRECTORY: Path = Path("X:\Track Analysis\output")

DATA_DIRECTORY: Path = Path("X:\Track Analysis\data")
CACHE_DIRECTORY: Path = Path("X:\Track Analysis\cache")

DEBUG: bool = True
VERBOSE: bool = True

MINIMUM_FUZZY_CONFIDENCE: float = 95  # 0-100
