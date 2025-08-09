import os
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict

import psutil

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature

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
EXPENSIVE_CACHE_DIRECTORY: Path = Path(r"N:/")
BENCHMARK_DIRECTORY: Path = Path(r"X:\Track Analysis\track_analysis\benchmarks")

ENERGY_CALCULATION_REGENERATE_LIBRARY_GROWTH_PERC: float = 0.1
CURRENT_ENERGY_TRAINING_VERSION_TO_USE: int = -1 # Every training (including the first one increments.) Switch to -1 when starting a new feature set.
NUMBER_OF_MFCCS: int = 20

DEBUG: bool = True
VERBOSE: bool = False
PROFILE_DATA_LOADING: bool = False
EMBED_BATCH_SIZE: int = 1500

CLEAR_CACHE: bool = False  # For developing purposes, clears the cache each time to reset -- for profiling.
DELETE_FINAL_DATA_BEFORE_START: bool = True  # For developing purposes; removes `enriched_scrobbles.csv` on start.
TEST_CACHE_BUILDER_MODE: bool = False  # For developing purposes; auto accepts every scrobble match to see data in cache clearly.

TEST_SAMPLE_SIZE: int | None = None  # The number of scrobbles to load (only applies to scrobbles, the library itself will always be fully loaded)

NO_MATCH_LABEL: str = "<NO MATCH>"

PHYSICAL_CPU_COUNT: int = psutil.cpu_count(logical=False)
LOGICAL_CPU_COUNT: int = psutil.cpu_count(logical=True)
NUM_WORKERS_CPU_HEAVY = LOGICAL_CPU_COUNT + 10
MAX_NEW_TRACKS_PER_RUN = 50

# ======================================================================================================================

KEYS_TO_BE_IGNORED_IN_CACHE_CHECK: List[str] = [
    "nujabes||luv sic hexalogy||luv sic pt2",
    "max richter||the blue notebooks 15 years||on the nature of daylight"
]

# ======================================================================================================================

class MFCCType(Enum):
    MEANS = auto()
    STDS = auto()
    VELOCITY_MEANS = auto()
    VELOCITY_STDS = auto()
    ACCELERATION_MEANS = auto()
    ACCELERATION_STDS = auto()

# Calculated - leave alone.
MFFC_LABEL_PREFIXES: Dict[MFCCType, str] = {
    MFCCType.MEANS: "mfcc_mean_",
    MFCCType.STDS: "mfcc_std_",
    MFCCType.VELOCITY_MEANS: "velocity_mean_",
    MFCCType.VELOCITY_STDS: "velocity_std_",
    MFCCType.ACCELERATION_MEANS: "acceleration_mean_",
    MFCCType.ACCELERATION_STDS: "acceleration_std_"
}

_MFCC_TYPE_TO_AUDIO_DATA_FEATURE: Dict[MFCCType, AudioDataFeature] = {
    MFCCType.MEANS: AudioDataFeature.MFCC_MEANS,
    MFCCType.STDS: AudioDataFeature.MFCC_STDS,
    MFCCType.VELOCITY_MEANS: AudioDataFeature.MFCC_VELOCITIES_MEANS,
    MFCCType.VELOCITY_STDS: AudioDataFeature.MFCC_VELOCITIES_STDS,
    MFCCType.ACCELERATION_MEANS: AudioDataFeature.MFCC_ACCELERATIONS_MEANS,
    MFCCType.ACCELERATION_STDS: AudioDataFeature.MFCC_ACCELERATIONS_STDS,
}

MFCC_LABEL_PREFIX_TO_AUDIO_DATA_FEATURE: Dict[str, AudioDataFeature] = {
    MFFC_LABEL_PREFIXES[prefix]: feature
    for prefix, feature in _MFCC_TYPE_TO_AUDIO_DATA_FEATURE.items()
}
