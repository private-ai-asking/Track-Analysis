from pathlib import Path

import pandas as pd
from pandas import DataFrame

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.scrobbling.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.scrobble_matcher import ScrobbleMatcher


class ScrobbleLinkerService:
    """Used to link scrobbles with library-level data using fuzzy matching. High-level API."""

    def __init__(self,
                 logger: HoornLogger,
                 library_data_path: Path,
                 scrobble_data_path: Path,
                 string_utils: StringUtils):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleLinker"
        self._string_utils: StringUtils = string_utils
        self._scrobble_data_loader: ScrobbleDataLoader = ScrobbleDataLoader(logger, library_data_path, scrobble_data_path, self._string_utils)
        self._scrobble_matcher: ScrobbleMatcher = ScrobbleMatcher(logger)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def _log_unmatched_amount(self, enriched_scrobble_date: pd.DataFrame) -> None:
        # Log unmatched count
        unmatched: int = enriched_scrobble_date["track_uuid"].isna().sum()
        total: int = len(enriched_scrobble_date)
        self._logger.info(
            f"Linked {total - unmatched} of {total} scrobbles. {unmatched} remain unmatched.",
            separator=self._separator
        )

    def link_scrobbles(self, output_path: Path, threshold: float = 90.0) -> None:
        """Links scrobble data to library data by matching tracks and writing the associated Track ID
        (if any) into the enriched scrobble data csv."""
        library_data: DataFrame = self._scrobble_data_loader.get_library_data()
        scrobble_data: DataFrame = self._scrobble_data_loader.get_scrobble_data()

        enriched_scrobble_data: DataFrame = self._scrobble_matcher.link_scrobbles(library_data, scrobble_data, threshold)
        self._log_unmatched_amount(enriched_scrobble_data)
        enriched_scrobble_data.to_csv(output_path, index=False)
