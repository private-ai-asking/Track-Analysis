import numpy as np
import pandas as pd

from track_analysis.components.track_analysis.constants import MFFC_LABEL_PREFIXES, MFCCType
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class MfccDFBuilder:
    def build(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """Builds the MFCC DataFrame from the raw results."""
        mfcc_means_col, mfcc_stds_col = AudioDataFeature.MFCC_MEANS.name, AudioDataFeature.MFCC_STDS.name
        velocity_means_col, velocity_stds_col = AudioDataFeature.MFCC_VELOCITIES_MEANS.name, AudioDataFeature.MFCC_VELOCITIES_STDS.name
        acceleration_means_col, acceleration_stds_col = AudioDataFeature.MFCC_ACCELERATIONS_MEANS.name, AudioDataFeature.MFCC_ACCELERATIONS_STDS.name

        if full_df.empty:
            return pd.DataFrame()

        for col in [mfcc_means_col, mfcc_stds_col, velocity_means_col, velocity_stds_col, acceleration_means_col, acceleration_stds_col]:
            if col not in full_df.columns:
                return pd.DataFrame()

        mfcc_data = {Header.UUID.value: full_df[Header.UUID.value]}
        mfcc_means_stacked = np.vstack(full_df[mfcc_means_col])
        mfcc_stds_stacked = np.vstack(full_df[mfcc_stds_col])
        velocity_means_stacked = np.vstack(full_df[velocity_means_col])
        velocity_stds_stacked = np.vstack(full_df[velocity_stds_col])
        acceleration_means_stacked = np.vstack(full_df[acceleration_means_col])
        acceleration_stds_stacked = np.vstack(full_df[acceleration_stds_col])

        for i in range(mfcc_means_stacked.shape[1]):
            mfcc_data[f"{MFFC_LABEL_PREFIXES[MFCCType.MEANS]}{i}"] = mfcc_means_stacked[:, i]
            mfcc_data[f"{MFFC_LABEL_PREFIXES[MFCCType.STDS]}{i}"] = mfcc_stds_stacked[:, i]
            mfcc_data[f"{MFFC_LABEL_PREFIXES[MFCCType.VELOCITY_MEANS]}{i}"] = velocity_means_stacked[:, i]
            mfcc_data[f"{MFFC_LABEL_PREFIXES[MFCCType.VELOCITY_STDS]}{i}"] = velocity_stds_stacked[:, i]
            mfcc_data[f"{MFFC_LABEL_PREFIXES[MFCCType.ACCELERATION_MEANS]}{i}"] = acceleration_means_stacked[:, i]
            mfcc_data[f"{MFFC_LABEL_PREFIXES[MFCCType.ACCELERATION_STDS]}{i}"] = acceleration_stds_stacked[:, i]

        return pd.DataFrame(mfcc_data)
