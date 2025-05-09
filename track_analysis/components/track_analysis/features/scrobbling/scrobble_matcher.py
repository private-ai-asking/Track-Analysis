from typing import List, Optional, Hashable

import pandas as pd
from pandas import DataFrame

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import calculate_similarity_score


class ScrobbleMatcher:
    """Used to match individual scrobbles to library data."""
    def __init__(self, logger: HoornLogger):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleMatcher"

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def _calculate_scrobble_match_score(self, scrobble_row: pd.Series, lib_row: pd.Series,
                                        w_title=0.5, w_artist=0.3, w_album=0.2) -> float:
        """Compute a weighted similarity score between a scrobble and library row."""
        field_weights = {
            "_n_title": w_title,
            "_n_artist": w_artist,
            "_n_album": w_album
        }

        rows: List[pd.Series] = [scrobble_row, lib_row]

        try:
            composite_score = calculate_similarity_score(rows, field_weights)
            titles = ", ".join([row["_n_title"] for row in rows])
            self._logger.trace(
                f"Calculated composite score: [{composite_score}] for titles [{titles}]",
                separator=self._separator
            )
            return composite_score
        except Exception as e:
            self._logger.error(f"Error calculating composite score: {str(e)}", separator=self._separator)
            return 0.0

    def _get_candidates(self, row: pd.Series, lib_blocks: dict, library_data: pd.DataFrame) -> pd.DataFrame:
        """Return candidate library rows by exact artist block, or all if missing."""
        self._logger.debug(f"Getting candidates for scrobble [{row['_n_title']}]...", separator=self._separator)

        artist = row["_n_artist"]
        candidates: pd.DataFrame = lib_blocks.get(artist, library_data)

        self._logger.debug(f"Done getting candidates for scrobble [{row['_n_title']}].", separator=self._separator)

        return candidates

    def _match_uuid(self, row: pd.Series, lib_blocks: dict, library_data: pd.DataFrame, threshold: float = 90.0) -> Optional[str]:
        """Find best matching library UUID for a scrobble row or None if below threshold."""
        candidates = self._get_candidates(row, lib_blocks, library_data)
        best_uuid = None
        best_score = 0.0
        for _, lib_row in candidates.iterrows():
            score = self._calculate_scrobble_match_score(row, lib_row)
            if score > best_score:
                best_score, best_uuid = score, lib_row["UUID"]
        return best_uuid if best_score >= threshold else None

    def link_scrobbles(self, library_data: pd.DataFrame, scrobble_data: pd.DataFrame, threshold: float = 95.0) -> pd.DataFrame:
        """
        Loads library and scrobble CSVs, normalizes fields, blocks on artist,
        computes fuzzy matches, and writes enriched scrobbles with 'track_uuid'.
        """

        # Block library by normalized artist for faster lookup
        lib_blocks: dict[Hashable, DataFrame] = {
            artist: group.reset_index(drop=True)
            for artist, group in library_data.groupby("_n_artist")
        }

        # Match each scrobble to a library UUID
        scrobble_data["track_uuid"] = scrobble_data.apply(
            lambda row: self._match_uuid(row, lib_blocks, library_data, threshold), axis=1
        )

        return scrobble_data
