from pathlib import Path
from typing import List

import pydantic

from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header


class AudioInfo(pydantic.BaseModel):
    path: Path
    metadata: List[AudioMetadataItem]

    def get_album_title(self) -> str:
        for metadata_item in self.metadata:
            if metadata_item.header == Header.Album:
                return metadata_item.value

    def get_track_title(self) -> str:
        for metadata_item in self.metadata:
            if metadata_item.header == Header.Title:
                return metadata_item.value

    def get_track_artist(self) -> str:
        for metadata_item in self.metadata:
            if metadata_item.header == Header.Artists:
                return metadata_item.value
