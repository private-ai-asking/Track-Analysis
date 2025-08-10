import re
import shutil
from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.track_downloading.utils.metadata_manipulator import \
    MetadataManipulator, MetadataKey
from track_analysis.components.track_analysis.features.track_downloading.utils.missing_metadata_finder import \
    MissingMetadataFinder
from track_analysis.components.track_analysis.features.track_downloading.model.recording_model import RecordingModel


class LibraryFileHandler:
    """Wrapper class around the low-level file handler for use with music libraries."""
    def __init__(self, logger: HoornLogger, supported_music_extensions: List[str]):
        self._logger = logger

        self._supported_music_extensions = supported_music_extensions
        self._separator = self.__class__.__name__

        self._file_handler = FileHandler()
        self._metadata_manipulator = MetadataManipulator(logger)
        self._missing_metadata_finder: MissingMetadataFinder = MissingMetadataFinder(logger)

    def get_music_files(self, directory: Path) -> List[Path]:
        """
        Returns a list of all music files in the specified directory.
        """
        files: List[Path] = []

        self._logger.trace("Searching for files with extensions '{}'.".format(self._supported_music_extensions), separator=self._separator)
        files.extend(self._file_handler.get_children_paths_fast(directory, self._supported_music_extensions, recursive=True))

        return files

    def organize_music_files(self, directory_path: Path, organized_path: Path):
        """
        Organizes the given music files into the specified organized_path.
        """
        all_files = [RecordingModel(metadata=self._metadata_manipulator.get_all_metadata(file), path=file) for file in self.get_music_files(directory_path)]
        missing_metadata_files = self._missing_metadata_finder.find_missing_metadata(all_files)
        correct_metadata_files = [file for file in all_files if file not in missing_metadata_files]

        for file in correct_metadata_files:
            self._place_accurate_file(file.path, file, organized_path)

        for file in missing_metadata_files:
            self._place_inaccurate_file(file.path, organized_path)

        self._remove_empty_directories(directory_path)
        self._remove_empty_directories(organized_path)

    def _place_accurate_file(self, file: Path, recording_model: RecordingModel, organized_path: Path) -> None:
        """
        Places a music file into an organized directory structure based on its metadata.

        Args:
            file (Path): The path to the music file.
            recording_model (RecordingModel): The metadata associated with the recording.
            organized_path (Path): The root path of the organized music library.
        """

        metadata = recording_model.metadata

        # Construct the new file name
        track_number = int(metadata[MetadataKey.TrackNumber])
        artist = metadata[MetadataKey.Artist]
        title = metadata[MetadataKey.Title]
        new_name = f"{track_number:02d} - {artist} - {title}{file.suffix.lower()}"
        new_name = self._clean_filename(new_name)

        # Construct the new directory path
        genre = metadata[MetadataKey.Genre].split(';')[0]
        album = metadata[MetadataKey.Album].replace("/", "-").replace(":", "_")
        new_path = organized_path / "SORTED" / genre / album / new_name

        if file == new_path:
            return  # File already exists at the correct location

        # Create the directory structure if it doesn't exist
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Move and rename the file
        shutil.move(file, new_path)
        self._logger.info(f"Moved '{file.name}' to '{new_path.parent.name}/{new_path.name}'")

    def _place_inaccurate_file(self, file: Path, organized_path: Path) -> None:
        cleaned_name = self._clean_filename(file.name)

        new_path: Path = organized_path.joinpath("_MISSING METADATA").joinpath(cleaned_name)

        if file == new_path:
            return  # File already exists at the incorrect location

        # Make directories if necessary
        new_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(file, new_path)
        self._logger.info(f"Moved {file.name} to {new_path.parent.name}/{new_path.name}")

    def recheck_missing_metadata(self, organized_path: Path):
        self.organize_music_files(organized_path.joinpath("_MISSING METADATA"), organized_path)

    def rescan_entire_library(self, organized_path):
        self.organize_music_files(organized_path, organized_path)

    def _remove_empty_directories(self, directory: Path) -> None:
        """
        Removes empty directories from the given directory and its subdirectories.
        """
        for child in directory.iterdir():
            if child.is_dir() and not child.iterdir():
                self._logger.debug(f"Removing empty directory: {child.name}")
                child.rmdir()

    def _clean_filename(self, filename: str, replacement_char='_') -> str:
        """
        Cleans a filename by removing unsupported characters and replacing them
        with a specified character (default: '_').

        Args:
          filename: The filename to clean.
          replacement_char: The character to replace unsupported characters with.

        Returns:
          The cleaned filename.
        """
        # Define a regular expression to match invalid characters
        invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'

        # Replace invalid characters with the replacement character
        cleaned_filename = re.sub(invalid_chars, replacement_char, filename)

        # Remove leading and trailing spaces and dots
        cleaned_filename = cleaned_filename.strip(' .')

        # Ensure the filename is not empty
        if not cleaned_filename:
            cleaned_filename = replacement_char

        return cleaned_filename
