from typing import List, Optional, Hashable

import pandas as pd
from pandas import DataFrame

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import calculate_similarity_score


class ScrobbleMatcher:
    """Used to match individual scrobbles to library data, with caching."""

    def __init__(self, logger: HoornLogger, cache_builder: CacheBuilder, key_combo: str = "||"):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleMatcher"
        self._cache_builder: CacheBuilder = cache_builder
        self._key_combo: str = key_combo
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
        artist = row["_n_artist"]
        candidates: DataFrame = lib_blocks.get(artist, library_data)
        return candidates

    def _make_cache_key(self, row: pd.Series) -> str:
        """Constructs a unique cache key based on normalized artist, title, and album."""
        return f"{row['_n_artist']}{self._key_combo}{row['_n_album']}{self._key_combo}{row['_n_title']}"

    def _match_uuid(self, row: pd.Series, lib_blocks: dict, library_data: pd.DataFrame, threshold: float = 95.0) -> Optional[str]:
        """Find best matching library UUID for a scrobble row or None if below threshold, using cache."""
        # Build cache key
        key = self._make_cache_key(row)
        # Check cache first
        cached = self._cache_builder.get(key)
        if cached is not None:
            self._logger.trace(f"Cache hit for key [{key}]: {cached}", separator=self._separator)
            return cached

        # No cached result: compute match
        candidates = self._get_candidates(row, lib_blocks, library_data)
        best_uuid = None
        best_score = 0.0
        for _, lib_row in candidates.iterrows():
            score = self._calculate_scrobble_match_score(row, lib_row)
            if score > best_score:
                best_score, best_uuid = score, lib_row["UUID"]
        # Apply threshold
        result = best_uuid if best_score >= threshold else None
        # Store in cache (even None to avoid re-compute)
        try:
            normalized_result = result if result is not None else "<NO ASSOCIATED KEY>"

            self._cache_builder.set(key, normalized_result)
            self._logger.trace(f"Cache set for key [{key}]: {result}", separator=self._separator)
        except Exception as e:
            self._logger.error(f"Failed to set cache for key [{key}]: {e}", separator=self._separator)
        return result

    def link_scrobbles(self, library_data: pd.DataFrame, scrobble_data: pd.DataFrame, threshold: float = 95.0) -> pd.DataFrame:
        """
        Blocks library by artist, matches scrobbles to library UUIDs using fuzzy matching,
        and caches results for speed on duplicates.
        """
        # Block library by normalized artist
        lib_blocks: dict[Hashable, DataFrame] = {
            artist: group.reset_index(drop=True)
            for artist, group in library_data.groupby("_n_artist")
        }
        # Match each scrobble to a library UUID
        scrobble_data["track_uuid"] = scrobble_data.apply(
            lambda row: self._match_uuid(row, lib_blocks, library_data, threshold), axis=1
        )
        return scrobble_data
