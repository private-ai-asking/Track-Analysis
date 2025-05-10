from pathlib import Path
from typing import Union, Dict

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class ScrobbleDataLoader:
    """Handles the loading of data related to scrobble analysis and linking."""
    def __init__(self,
                 logger: HoornLogger,
                 library_data_path: Path,
                 scrobble_data_path: Path,
                 string_utils: StringUtils,
                 scrobble_utils: ScrobbleUtility
                 ):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleDataLoader"
        self._string_utils: StringUtils = string_utils

        self._library_data_path: Path = library_data_path
        self._scrobble_data_path: Path = scrobble_data_path

        self._scrobble_utils: ScrobbleUtility = scrobble_utils

        self._loaded: bool = False

        self._lookup_cache: Dict[str, str] = {}

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def load(self, sample_rows: int = None) -> None:
        """Loads the data and normalizes it.

        Set sample_rows to None to disable and load everything. Only applies to scrobble data.
        """
        if self._loaded:
            self._logger.info("Data is already loaded. Skipping.", separator=self._separator)
            return

        self._load_data(self._library_data_path, self._scrobble_data_path, sample_rows=sample_rows)
        self._normalize_data()
        self._build_lookup()
        self._loaded = True

    def get_direct_lookup(self) -> Union[Dict[str, str], None]:
        """Builds a direct lookup between the library keycombo and its associated ID."""
        if not self._loaded:
            self._logger.warning("You haven't loaded the data yet!", separator=self._separator)
            return None

        return self._lookup_cache

    def get_library_data(self) -> Union[pd.DataFrame, None]:
        if not self._loaded:
            self._logger.warning("You haven't loaded the data yet!", separator=self._separator)
            return None

        return self._library_data

    def get_scrobble_data(self) -> Union[pd.DataFrame, None]:
        if not self._loaded:
            self._logger.warning("You haven't loaded the data yet!", separator=self._separator)
            return None

        return self._scrobble_data

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

    def _normalize_data(self) -> None:
        # Normalize text fields
        for df in (self._library_data, self._scrobble_data):
            df["_n_title"] = df["Title"].map(self._string_utils.normalize_field)
            df["_n_artist"] = df["Artist(s)"].map(self._string_utils.normalize_field)
            df["_n_album"] = df["Album"].map(self._string_utils.normalize_field)

    def _build_lookup(self):
        def __add_to_cache(row: pd.Series):
            self._lookup_cache[self._scrobble_utils.compute_key(row["_n_title"], row["_n_artist"], row["_n_album"])] = row["UUID"]

        self._library_data.apply(axis=1, func=lambda row: __add_to_cache(row))
