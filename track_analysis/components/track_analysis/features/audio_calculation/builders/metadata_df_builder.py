import uuid
from pathlib import Path
from typing import List

import pandas as pd

from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor


class MetadataDFBuilder:
    def __init__(self, file_handler: AudioFileHandler, tag_extractor: TagExtractor):
        self._file_handler = file_handler
        self._tag_extractor = tag_extractor

    def build_metadata_df(self, paths: List[Path], infos: List[AudioStreamsInfoModel], album_costs: List[AlbumCostModel]) -> pd.DataFrame:
        """Creates a DataFrame with file paths, UUIDs, and extracted tags."""
        records = []
        for path, info in zip(paths, infos):
            record = {
                Header.Audio_Path.value: path,
                Header.UUID.value: str(uuid.uuid4()),
                Header.Duration.value: info.duration,
                Header.Bitrate.value: info.bitrate,
                Header.Bit_Depth.value: info.bit_depth,
                Header.Max_Data_Per_Second.value: info.max_data_per_second_kbps,
                Header.Actual_Data_Rate.value: info.actual_data_rate_kbps,
                Header.Efficiency.value: info.efficiency,
                Header.Sample_Rate.value: info.sample_rate_kHz,
            }
            self._tag_extractor.add_extracted_metadata_to_track(record, info, album_costs)
            records.append(record)
        return pd.DataFrame(records)
