from pathlib import Path
from typing import Union, List

import mutagen
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


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

        file = mutagen.File(str(file_path), easy=True)
        self._logger.trace(f"Successfully loaded file: {file_path}.", separator=self._module_separator)

        if not file:
            self._logger.error(f"Error loading file {file_path}.")

        return file

    @staticmethod
    def _get_artists(file: mutagen.File) -> List[str]:
        artists = file.get('artists', "Unknown")
        if artists == "Unknown":
            artists = file.get('artist', "Unknown")
        if artists == "Unknown":
            artists = file.get('albumartist', ["Unknown"])

        return artists

    def add_extracted_metadata_to_track(self, track: pd.Series, audio_info: AudioStreamsInfoModel) -> None:
        """
        Extracts mp3 tags from the given audio file.

        Args:
            track (pd.Series): The data generation track.

        Returns:
            AudioMetadataItem: An object containing the extracted mp3 tags.
        """
        self._logger.trace(f"Extracting tags from {track}", separator=self._module_separator)

        track_path: Path = Path(track[Header.Audio_Path.value])
        file: mutagen.File = self._load_file(track_path)

        artists = self._get_artists(file)
        album_artists = file.get('albumartist', ["Unknown"])

        release_date = file.get('originaldate', ["Unknown"])[0]
        release_year = file.get('originalyear', ["Unknown"])[0] if (not release_date or release_date == "Unknown") else release_date[:4]

        # Basic Metadata
        track[Header.Title.value] = file.get('title', ["Unknown"])[0]
        track[Header.Album.value] = file.get('album', ["Unknown"])[0]
        track[Header.Artists.value] = artists
        track[Header.Album_Artists.value] = album_artists
        track[Header.Label.value] = file.get('label', ["Unknown"])[0]

        track[Header.Release_Year.value] = release_year
        track[Header.Release_Date.value] = release_date

        track[Header.Genre.value] = file.get('genre', ["Unknown"])[0]
        track[Header.Bought.value] = True if "[01] hq" in str(track_path) else False
        track[Header.Extension.value] = track_path.suffix

        file['bpm'] = str(round(audio_info.tempo, 4))
        file.save()

        self._logger.trace(f"Finished extracting tags from {track}", separator=self._module_separator)
        return None
