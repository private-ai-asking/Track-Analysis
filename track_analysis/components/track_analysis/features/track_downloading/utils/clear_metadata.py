from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.track_downloading.utils.library_file_handler import LibraryFileHandler
from track_analysis.components.track_analysis.features.track_downloading.utils.metadata_manipulator import \
    MetadataManipulator, MetadataKey


class ClearMetadata:
    """A tool to help clear metadata from music files."""
    def __init__(self, logger: HoornLogger, library_file_handler: LibraryFileHandler, metadata_manipulator: MetadataManipulator):
        self._logger = logger
        self._library_helper: LibraryFileHandler = library_file_handler
        self._metadata_helper: MetadataManipulator = metadata_manipulator

    def clear_genres(self, music_directory: Path):
        music_files = self._library_helper.get_music_files(music_directory)

        for music_file_path in music_files:
            self._logger.debug(f"Clearing genre from {music_file_path.stem}")
            self._metadata_helper.clear_metadata(music_file_path, MetadataKey.Genre, "No Genre")
            self._logger.info(f"Genre cleared from {music_file_path.stem}")

    def clear_dates(self, music_directory: Path):
        music_files = self._library_helper.get_music_files(music_directory)

        for music_file_path in music_files:
            self._logger.debug(f"Clearing dates from {music_file_path.stem}")
            self._metadata_helper.clear_metadata(music_file_path, MetadataKey.Date, "0000-00-00")
            self._logger.info(f"Dates cleared from {music_file_path.stem}")


