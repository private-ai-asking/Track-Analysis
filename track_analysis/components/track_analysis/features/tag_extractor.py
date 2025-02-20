from pathlib import Path
from typing import Union, List

import mutagen

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem


class TagExtractor:
    """
    Utility to extract mp3 tags from an audio file.
    """

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._module_separator: str = "TagExtractor"
        self._logger.trace("Successfully initialized TagExtractor.", separator=self._module_separator)

    def _load_file(self, file_path: Path) -> Union[mutagen.File, None]:
        self._logger.trace(f"Loading file from: {file_path}.", separator=self._module_separator)

        try:
            file = mutagen.File(str(file_path))
            self._logger.trace(f"Successfully loaded file: {file_path}.", separator=self._module_separator)
            return file
        except mutagen.MutagenError as e:
            self._logger.error(f"Error loading file {file_path}: {e}")
            return None

    def extract(self, audio_file: Path) -> AudioInfo:
        """
        Extracts mp3 tags from the given audio file.

        Args:
            audio_file (Path): The path to the audio file.

        Returns:
            AudioMetadataItem: An object containing the extracted mp3 tags.
        """
        self._logger.trace(f"Extracting mp3 tags from {audio_file}", separator=self._module_separator)

        metadata: List[AudioMetadataItem] = []

        file: mutagen.File = self._load_file(audio_file)

        # Basic Metadata
        metadata.append(AudioMetadataItem(header="Title", description="The track title.", value=file.get('title', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Album", description="The album where this track is part of.", value=file.get('album', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Artist(s)", description="The track artists.", value=file.get('artist', "Unknown")))
        metadata.append(AudioMetadataItem(header="Album Artist(s)", description="The album artists.", value=file.get('albumartist', "Unknown")))
        metadata.append(AudioMetadataItem(header="Label", description="The label associated with the track.", value=file.get('label', ["Unknown"])[0]))

        metadata.append(AudioMetadataItem(header="Release Year", description="The original year this track was released.", value=file.get('originalyear', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Release Date", description="The original date this track was released.", value=file.get('originaldate', ["Unknown"])[0]))

        # Sonic Metadata
        metadata.append(AudioMetadataItem(header="BPM", description="The tempo of the track.", value=file.get('bpm', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Energy Level", description="The energy level of the track.", value=file.get('energylevel', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Key", description="The camelot key of the track.", value=file.get('initialkey', ["Unknown"])[0]))

        self._logger.trace(f"Finished extracting mp3 tags from {audio_file}", separator=self._module_separator)
        return AudioInfo(metadata=metadata)
