import pickle
from pathlib import Path
from typing import Union, Dict, Optional, List

import faiss
import pandas as pd
from dateutil import parser, tz

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
                 scrobble_utils: ScrobbleUtility,
                 index_dir: Path,
                 keys_path: Path
                 ):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleDataLoader"
        self._string_utils: StringUtils = string_utils

        self._library_data_path: Path = library_data_path
        self._scrobble_data_path: Path = scrobble_data_path

        self._scrobble_utils: ScrobbleUtility = scrobble_utils

        self._index_dir: Path = index_dir
        self._keys_path = keys_path

        self._loaded: bool = False

        self._lookup_cache: Dict[str, str] = {}

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def library_by_uuid(self) -> Optional[Dict]:
        if not self._loaded:
            self._logger.warning("You haven't loaded the data yet!", separator=self._separator)
            return None

        return self._library_by_uuid

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

    def get_index(self) -> Optional[faiss.Index]:
        """Title, Artist, Album"""
        index_path: Path = Path(self._index_dir / "lib_combined.index")

        if not index_path.is_file():
            self._logger.warning(f"Couldn't load \"{index_path}\", since it does not exist!", separator=self._separator)
            return None

        return faiss.read_index(str(index_path))

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

    def get_scrobble_data(self, gold_standard: bool = False) -> Union[pd.DataFrame, None]:
        if not self._loaded:
            self._logger.warning("You haven't loaded the data yet!", separator=self._separator)
            return None

        if not gold_standard:
            return self._scrobble_data

        return self._gold_standard_data

    def _load_data(self, library_data_path: Path, scrobble_data_path: Path, sample_rows: int=None):
        self._logger.trace("Loading data...", separator=self._separator)

        # Load data
        self._library_data = pd.read_csv(library_data_path)
        self._scrobble_data = pd.read_csv(
            scrobble_data_path,
            names=["Scrobble Datetime", "Title", "Artist(s)", "Album", "Last.fm URL"],
            nrows=sample_rows,
            delimiter="\t",
            header=0
        )

        # self._gold_standard_data = pd.read_csv(
        #     scrobble_data_path,
        #     delimiter=","
        # )

        self._logger.debug("Successfully loaded data.", separator=self._separator)

    def _normalize_data(self) -> None:
        # Normalize text fields
        for df in (self._library_data, self._scrobble_data):
            df["_n_title"] = df["Title"].map(self._string_utils.normalize_field)
            df["_n_artist"] = df["Artist(s)"].map(self._string_utils.normalize_field)
            df["_n_album"] = df["Album"].map(self._string_utils.normalize_field)


        self._scrobble_data["Scrobble Datetime"] = self._scrobble_data["Scrobble Datetime"].map(self._localize_time_string)

    def _localize_time_string(self, utc_date_string: str) -> str:
        """
        Take an ISO‐style UTC timestamp string, localize it to the system's local timezone,
        and return a formatted local time string.
        """
        # 1. Parse the string into a datetime
        dt = parser.isoparse(utc_date_string)
        # 2. Ensure it's marked as UTC
        dt_utc = dt.replace(tzinfo=tz.UTC)
        # 3. Convert to local timezone
        local_tz = tz.tzlocal()
        dt_local = dt_utc.astimezone(local_tz)
        # 4. Return as formatted string
        return dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")


    def _build_lookup(self):
        def __add_to_cache(row: pd.Series):
            self._lookup_cache[self._scrobble_utils.compute_key(row["_n_title"], row["_n_artist"], row["_n_album"])] = row["UUID"]

        self._library_data.apply(axis=1, func=lambda row: __add_to_cache(row))

        self._library_by_uuid = (
            self._library_data
            .set_index("UUID", verify_integrity=True)   # optional: catch bad data early
            .to_dict(orient="index")
        )

    def get_keys(self) -> Optional[List[str]]:
        if not self._keys_path.is_file():
            self._logger.warning(f"Couldn't load \"{self._keys_path}\", since it does not exist!", separator=self._separator)
            return None

        with open(self._keys_path, 'rb') as f:
            return pickle.load(f)
