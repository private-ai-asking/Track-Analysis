import os
from pathlib import Path
from typing import List

ROOT: Path = Path(os.path.realpath(__file__)).parent.parent.parent.parent

FFMPEG_PATH: Path = Path(r"D:\[97] Installations\[01] Software\FFMPEG\bin\ffmpeg.exe")
COOKIES_FILE: Path = Path(r"D:\[98] PC Related\cookies.txt")
DOWNLOAD_CSV_FILE: Path = ROOT.joinpath("downloads.csv")
SUPPORTED_MUSIC_EXTENSIONS: List[str] = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".aiff", ".opus"]

ROOT_MUSIC_LIBRARY: Path = Path("W:\media\music\[02] organized") # NORMAL
# ROOT_MUSIC_LIBRARY: Path = Path("W:\\media\\music\\[03] test") # TESt
OUTPUT_DIRECTORY: Path = Path("X:\Track Analysis\output")

DATA_DIRECTORY: Path = Path("X:\Track Analysis\data")
CACHE_DIRECTORY: Path = Path("X:\Track Analysis\cache")
BENCHMARK_DIRECTORY: Path = Path(r"X:\Track Analysis\track_analysis\benchmarks")

DEBUG: bool = False
VERBOSE: bool = False
PROFILE_DATA_LOADING: bool = False
EMBED_BATCH_SIZE: int = 1500

CLEAR_CACHE: bool = False  # For developing purposes, clears the cache each time to reset -- for profiling.
DELETE_FINAL_DATA_BEFORE_START: bool = True  # For developing purposes; removes `enriched_scrobbles.csv` on start.
TEST_CACHE_BUILDER_MODE: bool = False  # For developing purposes; auto accepts every scrobble match to see data in cache clearly.

TEST_SAMPLE_SIZE: int = None  # The number of scrobbles to load (only applies to scrobbles, the library itself will always be fully loaded)

NO_MATCH_LABEL: str = "<NO MATCH>"

CPU_COUNT: int = 10
