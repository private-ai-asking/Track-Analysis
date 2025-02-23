from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class MatchPaths(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "MatchPathsPipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Matching paths...", separator=self._separator)

        cache: List[AudioInfo] = data.loaded_audio_info_cache
        generated_metadata: List[AudioInfo] = data.generated_audio_info

        updated_rows: List[AudioInfo] = []

        for cached_track_info in cache:
            cached_track_title = cached_track_info.get_track_title()
            cached_album_title = cached_track_info.get_album_title()
            cached_artist_name = cached_track_info.get_track_artist()

            for generated_track_info in generated_metadata:
                if (
                        generated_track_info.get_track_title() == cached_track_title and
                        generated_track_info.get_album_title() == cached_album_title and
                        generated_track_info.get_track_artist() == cached_artist_name
                ):
                    updated_metadata = []

                    for metadata_item_original in cached_track_info.metadata:
                        updated_metadata.append(metadata_item_original.model_copy(deep=True))

                    updated_metadata.append(AudioMetadataItem(header=Header.Audio_Path, description="The path of the track.", value=str(generated_track_info.path)))

                    updated_rows.append(AudioInfo(path=generated_track_info.path, metadata=updated_metadata))
                    self._logger.trace(f"Track match found: {generated_track_info.path}, metadata: {updated_metadata}", separator=self._separator)
                    break

        unmatched_rows: int = len(generated_metadata) - len(updated_rows)
        self._logger.warning(f"Number of unmatched tracks: {unmatched_rows}")

        data.loaded_audio_info_cache = updated_rows

        return data
