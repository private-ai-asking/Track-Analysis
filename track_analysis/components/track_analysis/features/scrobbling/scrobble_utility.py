from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class ScrobbleUtility:
    """Utility class for helpful misc methods relating to scrobble analysis."""
    def __init__(self, logger: HoornLogger, join_key: str = "||"):
        self._logger = logger
        self._separator = "ScrobbleUtility"

        self._join_key = join_key

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def compute_key(self, normalized_title: str, normalized_artists: str, normalized_album: str) -> str:
        combo_key: str = self._join_key
        return f"{normalized_artists}{combo_key}{normalized_album}{combo_key}{normalized_title}"
