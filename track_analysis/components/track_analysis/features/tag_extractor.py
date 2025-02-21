from pathlib import Path
from typing import Union, List
import mutagen

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header


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

    def _get_artists(self, file: mutagen.File) -> List[str]:
        artists = file.get('artists', "Unknown")
        if artists == "Unknown":
            artists = file.get('artist', "Unknown")
        if artists == "Unknown":
            artists = file.get('albumartist', ["Unknown"])

        return artists

    def extract(self, audio_file: Path) -> AudioInfo:
        """
        Extracts mp3 tags from the given audio file.

        Args:
            audio_file (Path): The path to the audio file.

        Returns:
            AudioMetadataItem: An object containing the extracted mp3 tags.
        """
        self._logger.trace(f"Extracting tags from {audio_file}", separator=self._module_separator)

        metadata: List[AudioMetadataItem] = []

        file: mutagen.File = self._load_file(audio_file)

        artists = self._get_artists(file)
        album_artists = file.get('albumartist', ["Unknown"])

        # Basic Metadata
        metadata.append(AudioMetadataItem(header=Header.Title, description="The track title.", value=file.get('title', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header=Header.Album, description="The album where this track is part of.", value=file.get('album', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header=Header.Artists, description="The track artists.", value=artists))
        metadata.append(AudioMetadataItem(header=Header.Album_Artists, description="The album artists.", value=album_artists))
        metadata.append(AudioMetadataItem(header=Header.Label, description="The label associated with the track.", value=file.get('label', ["Unknown"])[0]))

        metadata.append(AudioMetadataItem(header=Header.Release_Year, description="The original year this track was released.", value=file.get('originalyear', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header=Header.Release_Date, description="The original date this track was released.", value=file.get('originaldate', ["Unknown"])[0]))

        metadata.append(AudioMetadataItem(header=Header.Genre, description="The genre of the music.", value=file.get('genre', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header=Header.Bought, description="Whether the track was bought or not.", value=True if "[01] hq" in str(audio_file) else False))

        # Sonic Metadata
        metadata.append(AudioMetadataItem(header=Header.BPM, description="The tempo of the track.", value=file.get('bpm', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header=Header.Energy_Level, description="The energy level of the track.", value=file.get('energylevel', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header=Header.Key, description="The camelot key of the track.", value=file.get('initialkey', ["Unknown"])[0]))


        self._logger.trace(f"Finished extracting tags from {audio_file}", separator=self._module_separator)
        return AudioInfo(metadata=metadata, path=audio_file)
