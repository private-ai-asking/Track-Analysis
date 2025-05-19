from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.constants import VERBOSE
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.features.data_generation.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.features.data_generation.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import PipelineContextModel


class GetAudioMetadata(IPipe):
    def __init__(self, logger: HoornLogger, tag_extractor: TagExtractor, string_utils: StringUtils):
        self._separator = "GetAudioMetadataPipe"

        self._logger = logger
        self._tag_extractor = tag_extractor
        self._string_utils = string_utils
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Extracting all audio metadata...", separator=self._separator)
        audio_info: List[AudioInfo] = []

        processed: int = 0
        to_process: int = len(data.filtered_audio_file_paths)

        for track_path in data.filtered_audio_file_paths:
            audio_info.append(self._tag_extractor.extract(track_path))
            self._logger.trace(f"Extracted audio metadata for: {track_path}", separator=self._separator)

            if VERBOSE:
                self._logger.debug(f"Extracted audio metadata: {audio_info}", separator=self._separator)

            album_cost = 0

            for album_cost_info in data.album_costs:
                if audio_info[-1].get_album_title() == album_cost_info.Album_Title:
                    album_cost = album_cost_info.Album_Cost
                    break


            original_artists = ""

            for mdi in audio_info[-1].metadata:
                if mdi.header == Header.Artists:
                    original_artists = mdi.value

            primary_artist = self._string_utils.extract_primary_from_sequence(original_artists)

            audio_info[-1].metadata.append(AudioMetadataItem(header=Header.Album_Cost, description="The cost of the album this track is a part of.", value=album_cost))
            audio_info[-1].metadata.append(AudioMetadataItem(header=Header.Primary_Artist, description="", value=primary_artist))

            processed += 1
            self._logger.info(f"Successfully extracted metadata for track: {processed}/{to_process} ({round(processed/to_process * 100, 4)}%)", separator=self._separator)

        data.generated_audio_info = audio_info

        self._logger.info("Successfully extracted all audio metadata.", separator=self._separator)
        return data
