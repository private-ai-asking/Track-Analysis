import json
from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader


class UnmatchedLibraryTracker:
    """Utility class to see which library tracks do not have associated scrobbles in cache."""
    def __init__(self, logger: HoornLogger, data_loader: ScrobbleDataLoader, cache_path: Path):
        self._logger = logger
        self._separator = "UnmatchedLibraryTracker"
        self._loader = data_loader
        self._cache_path = cache_path

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def print_unmatched_tracks(self) -> None:
        # 1) load library
        self._loader.load()
        library_df = self._loader.get_library_data()
        if library_df is None:
            self._logger.error("Library data not available.", separator=self._separator)
            return

        # 2) load cache JSON
        if not self._cache_path.is_file():
            self._logger.warning(f"Cache file not found at {self._cache_path}", separator=self._separator)
            print("No cache file found; all library tracks are unmatched.")
            return

        try:
            with open(self._cache_path, 'r', encoding='utf-8') as f:
                cache_dict = json.load(f)
        except Exception as e:
            self._logger.error(f"Failed to read cache JSON: {e}", separator=self._separator)
            return

        # 3) collect all associated_uuid values
        used_uuids = {
            entry.get("associated_uuid")
            for entry in cache_dict.values()
            if entry.get("associated_uuid") is not None
        }

        # 4) filter library rows whose UUID not in used_uuids
        unmatched = library_df[~library_df["UUID"].isin(used_uuids)]

        # 5) print results
        if unmatched.empty:
            print("✅ All library tracks have at least one cached scrobble association.")
        else:
            count = len(unmatched)
            print(f"🔍 Found {count} library tracks with no cache association:\n")
            for _, row in unmatched.iterrows():
                uuid   = row["UUID"]
                title  = row.get("Title", "<no title>")
                artist = row.get("Artist(s)", "<no artist>")
                album  = row.get("Album", "<no album>")
                print(f"• {uuid} — {artist} / {album} / {title}")
