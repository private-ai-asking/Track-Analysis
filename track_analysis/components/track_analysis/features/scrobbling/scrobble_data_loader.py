from pathlib import Path
from typing import Union

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils


class ScrobbleDataLoader:
    """Handles the loading of data related to scrobble analysis and linking."""
    def __init__(self,
                 logger: HoornLogger,
                 library_data_path: Path,
                 scrobble_data_path: Path,
                 string_utils: StringUtils):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleDataLoader"
        self._string_utils: StringUtils = string_utils

        self._library_data_path: Path = library_data_path
        self._scrobble_data_path: Path = scrobble_data_path

        self._loaded: bool = False

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
        self._loaded = True

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
