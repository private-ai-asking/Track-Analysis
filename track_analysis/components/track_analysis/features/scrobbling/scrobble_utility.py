import pandas as pd

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class ScrobbleUtility:
    """Utility class for helpful misc methods relating to scrobble analysis."""
    def __init__(self, logger: HoornLogger, cache_builder: CacheBuilder, join_key: str = "||"):
        self._logger = logger
        self._separator = "ScrobbleUtility"

        self._cache: CacheBuilder = cache_builder
        self._join_key: str = join_key

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def compute_key(self, normalized_title: str, normalized_artists: str, normalized_album: str) -> str:
        combo_key: str = self._join_key
        return f"{normalized_artists}{combo_key}{normalized_album}{combo_key}{normalized_title}"

    def save_cache_item(self, key: str, uuid: str, confidence_factor_percentage: float, library_data_frame: pd.DataFrame) -> None:
        """Extract the relevant row and store it directly as a dict in cache."""
        row = library_data_frame.set_index('UUID').loc[uuid]

        payload = {
            'associated_uuid': uuid,
            'associated_track_title': row['Title'],
            'associated_track_album': row['Album'],
            'associated_track_artist': row['Artist(s)'],
            'confidence_factor_percentage': confidence_factor_percentage,
        }

        self._cache.set(key, payload)
        self._logger.debug(
            f"Accepted: {key} -> {uuid} (confidence={confidence_factor_percentage:.2f})",
            separator=self._separator,
        )
