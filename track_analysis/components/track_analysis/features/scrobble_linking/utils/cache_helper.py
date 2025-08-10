from typing import Dict

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_data_loader import ScrobbleDataLoader


class ScrobbleCacheHelper:
    def __init__(self, logger: HoornLogger, loader: ScrobbleDataLoader, cache_builder: CacheBuilder):
        self._logger = logger
        self._separator: str = "ScrobbleCacheHelper"

        self._loader = loader

        self._cache: CacheBuilder = cache_builder

    def save_cache_item(
            self,
            key: str,
            uuid: str,
            confidence_factor_percentage: float
    ) -> None:
        """Extracts the row for `uuid` from our pre-indexed library and caches it."""
        _library_by_uuid: Dict = self._loader.get_library_row_by_uuid_lookup()

        try:
            row = _library_by_uuid[uuid]
        except KeyError:
            self._logger.warning(f"UUID {uuid} not found in library.", separator=self._separator)
            return

        payload = {
            "associated_uuid": uuid,
            "associated_track_title":  row["Title"],
            "associated_track_album":  row["Album"],
            "associated_track_artist": row["Artist(s)"],
            "confidence_factor_percentage": confidence_factor_percentage,
        }

        self._cache.set(key, payload)
        self._logger.debug(
            f"Accepted: {key} -> {uuid} (confidence={confidence_factor_percentage:.2f})",
            separator=self._separator,
        )
