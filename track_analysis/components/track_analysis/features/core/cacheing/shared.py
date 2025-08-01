from joblib import Memory

from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY

MEMORY     = Memory(EXPENSIVE_CACHE_DIRECTORY, verbose=0, mmap_mode="r")
