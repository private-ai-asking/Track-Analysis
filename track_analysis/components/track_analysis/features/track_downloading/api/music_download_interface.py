from abc import abstractmethod
from typing import List

from track_analysis.components.track_analysis.features.track_downloading.model.download_model import DownloadModel


class MusicDownloadInterface:
    def __init__(self, is_child: bool = False):
        if is_child:
            return

        raise NotImplementedError("You cannot instantiate an interface. Use a concrete implementation.")

    @abstractmethod
    def download_tracks(self, tracks_to_download: List[DownloadModel]) -> List[DownloadModel]:
        """Downloads the given tracks."""
        raise NotImplementedError("You are attempting to call the method of an interface directly, use the concrete implementation.")
