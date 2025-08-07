from typing import List, Tuple

import pandas as pd

from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.helpers.key_extractor import KeyExtractionResult


class KeyDataFramesBuilder:
    def build(
            self, key_results: List[KeyExtractionResult], meta_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Processes a list of raw KeyExtractionResult objects into two DataFrames:
        1. A DataFrame with the main key info for each track.
        2. A list of DataFrames, one for each track's key progression.
        """
        main_key_records, key_progression_dfs = [], []

        for res in key_results:
            uuid = meta_df.iloc[res.index][Header.UUID.value]

            progression_df_for_track = pd.DataFrame([
                {
                    "Track UUID": uuid,
                    "Segment Start": lk.interval_start,
                    "Segment End": lk.interval_end,
                    "Segment Key": lk.key,
                } for lk in res.local_info
            ])
            key_progression_dfs.append(progression_df_for_track)

            main_key_records.append({
                'original_index': res.index,
                Header.Key.value: res.global_key,
                Header.Start_Key.value: res.local_info[0].key if res.local_info else None,
                Header.End_Key.value: res.local_info[-1].key if res.local_info else None,
            })

        return pd.DataFrame(main_key_records), key_progression_dfs
