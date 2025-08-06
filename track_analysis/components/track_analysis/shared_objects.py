from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import HDF5Memory

MEMORY     = HDF5Memory(EXPENSIVE_CACHE_DIRECTORY / "track_analysis_cache.h5")
