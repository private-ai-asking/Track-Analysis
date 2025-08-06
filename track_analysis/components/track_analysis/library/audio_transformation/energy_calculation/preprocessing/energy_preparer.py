from typing import List, Dict, Any, Optional

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING


class EnergyDataPreparer:
    """
    Handles the preparation of data for energy model training and prediction.
    Specifically, it manages the merging of core features with MFCC data.
    """
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._header_to_feature_map = {v.value: k for k, v in FEATURE_TO_HEADER_MAPPING.items()}

    def _get_standard_feature(self, feature_name: str, feature_data: Dict[AudioDataFeature, Any]) -> Optional[Any]:
        """Retrieves a standard feature's value from the data dictionary."""
        feature_enum = self._header_to_feature_map.get(feature_name)
        return feature_data.get(feature_enum) if feature_enum else None

    @staticmethod
    def _get_mfcc_feature(feature_name: str, feature_data: Dict[AudioDataFeature, Any]) -> Optional[float]:
        """Extracts a single MFCC value from its corresponding feature array."""
        try:
            mfcc_index = int(feature_name.split('_')[-1])
            feature_enum = AudioDataFeature.MFCC_MEANS if 'mean' in feature_name else AudioDataFeature.MFCC_STDS

            if feature_enum in feature_data and mfcc_index < len(feature_data[feature_enum]):
                return feature_data[feature_enum][mfcc_index]
        except (ValueError, IndexError):
            return None
        return None

    @staticmethod
    def prepare_for_prediction(df_to_process: pd.DataFrame,
                               model_features: List[str]) -> pd.DataFrame:
        """
        Prepares data for prediction by selecting required features and handling missing values.
        """
        prepared_df = df_to_process[model_features]

        if prepared_df.isnull().values.any():
            missing = prepared_df.columns[prepared_df.isnull().any()].tolist()
            raise ValueError(f"Incomplete feature data after preparation. Missing: {missing}")

        return prepared_df

    @staticmethod
    def prepare_for_training(training_data: pd.DataFrame, config) -> pd.DataFrame:
        feature_names = config.get_feature_names()
        missing_columns = [col for col in feature_names if col not in training_data.columns]
        if missing_columns:
            raise ValueError(f"The provided training_data is missing required columns: {missing_columns}")
        return training_data[feature_names].dropna()
