from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobble_linking.utils.cache_helper import ScrobbleCacheHelper
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_data_loader import \
    ScrobbleDataLoader


class CacheHelperFactory:
    """Creates the cache helper."""
    def __init__(self, logger: HoornLogger, scrobble_data_loader: ScrobbleDataLoader, cache_builder: CacheBuilder):
        self._logger = logger
        self._scrobble_data_loader = scrobble_data_loader
        self._cache_builder = cache_builder

    def create(self) -> ScrobbleCacheHelper:
        return ScrobbleCacheHelper(
            logger=self._logger,
            loader=self._scrobble_data_loader,
            cache_builder=self._cache_builder,
        )
