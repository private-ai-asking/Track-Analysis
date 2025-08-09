from typing import Dict

import pandas as pd

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class MainAudioInfoDFBuilder:
    def __init__(self, feature_to_header_map: Dict[AudioDataFeature, Header]):
        self._feature_to_header_map = feature_to_header_map

    def build(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """Builds the main audio info DataFrame by renaming and selecting columns."""
        if full_df.empty:
            return pd.DataFrame()

        rename_map = {
            feat.name: head.value
            for feat, head in self._feature_to_header_map.items()
            if feat.name in full_df.columns
        }
        main_df = full_df.rename(columns=rename_map)

        # Select only the columns defined in the Header enum
        final_columns = [h.value for h in Header if h.value in main_df.columns]
        return main_df[final_columns]
