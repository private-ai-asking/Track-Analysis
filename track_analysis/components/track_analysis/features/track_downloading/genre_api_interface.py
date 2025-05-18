from abc import abstractmethod

from track_analysis.components.track_analysis.features.track_downloading.model.genre_model import GenreDataModel


class GenreAPIInterface:
    def __init__(self, is_child: bool = False):
        if is_child:
            return

        raise NotImplementedError("You cannot instantiate an interface. Use a concrete implementation.")

    @abstractmethod
    def get_genre_data(self, track_title: str, track_artist: str = None, track_album: str = None, track_id: str = None, album_id: str = None) -> GenreDataModel:
        """Returns the genre data model."""
        raise NotImplementedError("You are attempting to call the method of an interface directly, use the concrete implementation.")
