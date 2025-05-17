from pathlib import Path
from typing import Union

import mutagen

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.features.data_generation.model.audio_metadata_item import \
    AudioMetadataItem
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.track_processor_interface import \
    ITrackProcessorStrategy


class BPMTrackProcessor(ITrackProcessorStrategy):
    def __init__(self, logger: HoornLogger):
        super().__init__(logger, "BPM", is_child=True)

    def _load_file(self, file_path: Path) -> Union[mutagen.File, None]:
        file = mutagen.File(str(file_path), easy=True)
        return file

    def process_track(self, track: AudioInfo) -> AudioInfo:
        """Processes a single track and returns the updated AudioInfo."""
        self._logger.trace(f"Processing track: {track.get_printed()}")

        file: mutagen.File = self._load_file(track.path)

        updated_metadata = []

        for metadata_item_original in track.metadata:
            if metadata_item_original.header == Header.BPM:
                continue

            updated_metadata.append(metadata_item_original.model_copy(deep=True))


        updated_metadata.append(AudioMetadataItem(header=Header.BPM, description="The tempo of the track.", value=file.get('bpm', ["Unknown"])[0]))

        self._logger.trace(f"Processed track: {track.get_printed()}")
        return AudioInfo(path=track.path, metadata=updated_metadata, timeseries_data=track.timeseries_data)
