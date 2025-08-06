import numpy as np
import pandas as pd

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class MfccDFBuilder:
    def build(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """Builds the MFCC DataFrame from the raw results."""
        mfcc_means_col, mfcc_stds_col = AudioDataFeature.MFCC_MEANS.name, AudioDataFeature.MFCC_STDS.name

        if full_df.empty or mfcc_means_col not in full_df.columns or mfcc_stds_col not in full_df.columns:
            return pd.DataFrame()

        mfcc_data = {Header.UUID.value: full_df[Header.UUID.value]}
        mfcc_means_stacked = np.vstack(full_df[mfcc_means_col])
        mfcc_stds_stacked = np.vstack(full_df[mfcc_stds_col])

        for i in range(mfcc_means_stacked.shape[1]):
            mfcc_data[f"mfcc_mean_{i}"] = mfcc_means_stacked[:, i]
            mfcc_data[f"mfcc_std_{i}"] = mfcc_stds_stacked[:, i]

        return pd.DataFrame(mfcc_data)
