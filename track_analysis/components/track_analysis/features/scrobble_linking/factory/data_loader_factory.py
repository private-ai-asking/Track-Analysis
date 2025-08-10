from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_utility import ScrobbleUtility


class ScrobbleDataLoaderFactory:
    """Creates a scrobble data loader."""
    def __init__(self,
                 logger: HoornLogger,
                 string_utils: StringUtils,
                 scrobble_utils: ScrobbleUtility,
                 library_data_path: Path,
                 scrobble_index_dir: Path,
                 library_keys_path: Path,
                 scrobble_data_path: Path):
        self._logger = logger
        self._string_utils = string_utils
        self._scrobble_utils = scrobble_utils

        self._library_data_path = library_data_path
        self._scrobble_index_dir = scrobble_index_dir
        self._library_keys_path = library_keys_path
        self._scrobble_data_path = scrobble_data_path

    def create(self) -> ScrobbleDataLoader:
        return ScrobbleDataLoader(
            self._logger,
            self._library_data_path,
            self._scrobble_data_path,
            self._string_utils,
            self._scrobble_utils,
            self._scrobble_index_dir,
            self._library_keys_path,
        )
