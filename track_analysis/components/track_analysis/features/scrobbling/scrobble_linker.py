import re
from pathlib import Path
from typing import Optional, Hashable

import pandas as pd
import unicodedata
from pandas import DataFrame
from rapidfuzz import fuzz

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class ScrobbleLinker:
    """Used to link scrobbles with library-level data using fuzzy matching."""

    def __init__(self, logger: HoornLogger, library_data_path: Path, scrobble_data_path: Path):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleLinker"

        self._load_data(library_data_path, scrobble_data_path, sample_rows=20)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def _load_data(self, library_data_path: Path, scrobble_data_path: Path, sample_rows: int=None):
        self._logger.trace("Loading data...", separator=self._separator)

        # Load data
        self._library_data = pd.read_csv(library_data_path)
        self._scrobble_data = pd.read_csv(
            scrobble_data_path,
            names=["Scrobble Datetime", "Title", "Artist(s)", "Album", "Last.fm URL"],
            nrows=sample_rows,
            delimiter="\t"
        )

        self._logger.debug("Successfully loaded data.", separator=self._separator)

    def _normalize_field(self, field_content: str) -> str:
        """Normalizes a field to lowercase, strips punctuation/accents, collapses whitespace."""
        if not isinstance(field_content, str):
            return ""

        self._logger.trace(f"Normalizing Field [{field_content}]", separator=self._separator)

        normalized_string = field_content.lower()
        normalized_string = unicodedata.normalize("NFKD", normalized_string)
        normalized_string = re.sub(r"’", "'", normalized_string)
        normalized_string = re.sub(r"[^a-z0-9 ]", " ", normalized_string)
        normalized_string = re.sub(r"\s+", " ", normalized_string).strip()

        return normalized_string

    def _composite_score(self, scrobble_row: pd.Series, lib_row: pd.Series,
                         w_title=0.5, w_artist=0.3, w_album=0.2) -> float:
        """Compute a weighted similarity score between a scrobble and library row."""
        title_score: float = fuzz.token_sort_ratio(scrobble_row["_n_title"], lib_row["_n_title"])
        artist_score: float = fuzz.token_sort_ratio(scrobble_row["_n_artist"], lib_row["_n_artist"])
        album_score: float = fuzz.token_sort_ratio(scrobble_row["_n_album"], lib_row["_n_album"])

        composite_score: float = (w_title * title_score) + (w_artist * artist_score) + (w_album * album_score)

        self._logger.trace(f"Calculated composite score: [{composite_score}] for [{scrobble_row['_n_title']}], [{scrobble_row['_n_album']}]", separator=self._separator)

        return composite_score

    def _get_candidates(self, row: pd.Series, lib_blocks: dict) -> pd.DataFrame:
        """Return candidate library rows by exact artist block, or all if missing."""
        self._logger.debug(f"Getting candidates for scrobble [{row['_n_title']}]...", separator=self._separator)

        artist = row["_n_artist"]
        candidates: pd.DataFrame = lib_blocks.get(artist, self._library_data)

        self._logger.debug(f"Done getting candidates for scrobble [{row['_n_title']}].", separator=self._separator)

        return candidates

    def _match_uuid(self, row: pd.Series, lib_blocks: dict, threshold: float = 90.0) -> Optional[str]:
        """Find best matching library UUID for a scrobble row or None if below threshold."""
        candidates = self._get_candidates(row, lib_blocks)
        best_uuid = None
        best_score = 0.0
        for _, lib_row in candidates.iterrows():
            score = self._composite_score(row, lib_row)
            if score > best_score:
                best_score, best_uuid = score, lib_row["UUID"]
        return best_uuid if best_score >= threshold else None

    def link_scrobbles(self, output_path: Path, threshold: float = 95.0) -> pd.DataFrame:
        """
        Loads library and scrobble CSVs, normalizes fields, blocks on artist,
        computes fuzzy matches, and writes enriched scrobbles with 'track_uuid'.
        """
        # Normalize text fields
        for df in (self._library_data, self._scrobble_data):
            df["_n_title"] = df["Title"].map(self._normalize_field)
            df["_n_artist"] = df["Artist(s)"].map(self._normalize_field)
            df["_n_album"] = df["Album"].map(self._normalize_field)

        # Block library by normalized artist for faster lookup
        lib_blocks: dict[Hashable, DataFrame] = {
            artist: group.reset_index(drop=True)
            for artist, group in self._library_data.groupby("_n_artist")
        }

        # Match each scrobble to a library UUID
        self._scrobble_data["track_uuid"] = self._scrobble_data.apply(
            lambda row: self._match_uuid(row, lib_blocks, threshold), axis=1
        )

        # Log unmatched count
        unmatched: int = self._scrobble_data["track_uuid"].isna().sum()
        total: int = len(self._scrobble_data)
        self._logger.info(
            f"Linked {total - unmatched} of {total} scrobbles. {unmatched} remain unmatched.",
            separator=self._separator
        )

        # Write enriched scrobbles
        self._scrobble_data.to_csv(output_path, index=False)
        return self._scrobble_data
