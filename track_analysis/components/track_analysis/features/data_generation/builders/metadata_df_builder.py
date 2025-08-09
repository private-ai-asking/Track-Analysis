import uuid
from pathlib import Path
from typing import List

import pandas as pd

from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor


class MetadataDFBuilder:
    def __init__(self, tag_extractor: TagExtractor):
        self._tag_extractor = tag_extractor

    def build_metadata_df(self, paths: List[Path], album_costs: List[AlbumCostModel]) -> pd.DataFrame:
        """Creates a DataFrame with file paths, UUIDs, and extracted tags."""
        records = []
        for path in paths:
            record = {
                Header.Audio_Path.value: path,
                Header.UUID.value: str(uuid.uuid4()),
            }
            self._tag_extractor.add_extracted_metadata_to_track(record, album_costs)
            records.append(record)
        return pd.DataFrame(records)
