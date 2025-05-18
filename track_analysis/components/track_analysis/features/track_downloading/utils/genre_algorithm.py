from typing import List

import musicbrainzngs

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.track_downloading.api.genre_api_interface import GenreAPIInterface
from track_analysis.components.track_analysis.features.track_downloading.model.genre_model import GenreDataModel
from track_analysis.components.track_analysis.features.track_downloading.api.music_brainz_genre_api import \
    MusicBrainzGenreAPI


class GenreAlgorithm:
    """
    Tool to help classify a song into different genres.
    Uses several APIs to come up with a somewhat accurate guess.
    """

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._genre_api_interfaces: List[GenreAPIInterface] = [
            MusicBrainzGenreAPI(logger)
        ]

    def get_genre_data(self, mbid: str, album_id: str = None) -> GenreDataModel:
        self._logger.debug(f"Getting genre data for {mbid}...")

        recording = musicbrainzngs.get_recording_by_id(mbid, includes=['artists'])
        title = recording['recording']['title']
        artist = recording['recording']['artist-credit'][0]['artist']['name']

        genres: List[GenreDataModel] = []
        for api in self._genre_api_interfaces:
            genres.append(api.get_genre_data(track_title=title, track_artist=artist, track_id=mbid, album_id=album_id))

        return genres[0]
