from pathlib import Path
from typing import Union

import mutagen

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.data_generation.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.features.data_generation.model.audio_metadata_item import \
    AudioMetadataItem
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.track_processor_interface import \
    ITrackProcessorStrategy


class PrimaryArtistProcessor(ITrackProcessorStrategy):
    def __init__(self, logger: HoornLogger, string_utils: StringUtils):
        super().__init__(logger, "PrimaryArtist", is_child=True)
        self._string_utils = string_utils

    def process_track(self, track: AudioInfo) -> AudioInfo:
        """Processes a single track and returns the updated AudioInfo."""
        self._logger.trace(f"Processing track: {track.get_printed()}")

        updated_metadata = []
        original_artists = ""

        for metadata_item_original in track.metadata:
            if metadata_item_original.header == Header.Primary_Artist:
                continue
            elif metadata_item_original.header == Header.Artists:
                original_artists = metadata_item_original.value

            updated_metadata.append(metadata_item_original.model_copy(deep=True))

        primary_artist = self._string_utils.extract_primary_from_sequence(original_artists)

        updated_metadata.append(AudioMetadataItem(header=Header.Primary_Artist, description="The primary artist.", value=primary_artist))

        self._logger.trace(f"Processed track: {track.get_printed()}")
        return AudioInfo(path=track.path, metadata=updated_metadata, timeseries_data=track.timeseries_data)
