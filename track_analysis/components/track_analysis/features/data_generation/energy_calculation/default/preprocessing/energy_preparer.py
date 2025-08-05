import pprint
from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class EnergyDataPreparer:
    """
    Handles the preparation of data for energy model training and prediction.
    Specifically, it manages the merging of core features with MFCC data.
    """
    def __init__(self, mfcc_data: pd.DataFrame, logger: HoornLogger):
        self._mfcc_data = mfcc_data
        self._logger = logger
        self._separator = self.__class__.__name__

    def prepare_for_training(self,
                             training_data: pd.DataFrame,
                             config: EnergyModelConfig) -> pd.DataFrame:
        """
        Prepares the final training data by merging MFCCs if specified in the config.
        """
        final_training_data = training_data.copy()

        if config.use_mfcc:
            mfcc_cols_to_join = [f for f in config.get_feature_names() if f.startswith('mfcc_')]
            final_training_data = self._merge_with_mfcc_data(
                main_df=final_training_data,
                columns_to_join=mfcc_cols_to_join
            )

        return final_training_data

    def prepare_for_prediction(self,
                               df_to_process: pd.DataFrame,
                               model_features: List[str]) -> pd.DataFrame:
        """
        Prepares data for prediction based on the required features of a loaded model.
        """
        mfcc_columns_to_join = [f for f in model_features if f.startswith("mfcc_")]
        if not mfcc_columns_to_join:
            return df_to_process.copy()

        prepared_df = self._merge_with_mfcc_data(
            main_df=df_to_process,
            columns_to_join=mfcc_columns_to_join
        )

        columns_to_return = model_features + [Header.UUID.value]

        final_columns = list(dict.fromkeys(columns_to_return))

        return prepared_df[final_columns]

    def _merge_with_mfcc_data(self, main_df: pd.DataFrame, columns_to_join: List[str]) -> pd.DataFrame:
        """Merges a dataframe with the stored MFCC data on the UUID index."""
        if self._mfcc_data is None or self._mfcc_data.empty:
            raise ValueError("MFCC data was not provided or is empty.")

        mfcc_df_indexed = self._mfcc_data.set_index(Header.UUID.value)

        available_mfcc_columns = [col for col in columns_to_join if col in mfcc_df_indexed.columns]
        missing_columns = set(columns_to_join) - set(available_mfcc_columns)

        if missing_columns:
            self._logger.warning(
                f"Missing MFCC columns: {pprint.pformat(missing_columns)}. "
                "These features will be treated as NaN.",
                separator=self._separator
            )

        mfcc_df_filtered = mfcc_df_indexed.loc[:, available_mfcc_columns]

        main_df_indexed = main_df.set_index(Header.UUID.value)
        merged_df = main_df_indexed.join(mfcc_df_filtered, how='left')

        for col in missing_columns:
            merged_df[col] = pd.NA

        return merged_df.reset_index()
