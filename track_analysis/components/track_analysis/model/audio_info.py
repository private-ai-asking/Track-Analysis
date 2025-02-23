from pathlib import Path
from typing import List, Optional

import pydantic

from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header


class AudioInfo(pydantic.BaseModel):
    path: Path
    metadata: List[AudioMetadataItem]
    timeseries_data: Optional[List["AudioInfo"]] = []

    def get_printed(self) -> str:
        return f"AudioInfo(track_title={self.get_track_title()}, path={self.path})"

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

    def get_track_id(self) -> str:
        for metadata_item in self.metadata:
            if metadata_item.header == Header.UUID:
                return metadata_item.value
