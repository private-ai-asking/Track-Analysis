from typing import Optional

import pandas as pd

from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class TrainingDataBuilder:
    """
    Handles the merging of main track data with MFCC data for training.
    """
    def build(self,
              main_df: pd.DataFrame,
              mfcc_df: Optional[pd.DataFrame],
              use_mfcc: bool) -> pd.DataFrame:
        """
        Builds the final training DataFrame.

        Args:
            main_df: The DataFrame containing the core audio features.
            mfcc_df: The DataFrame containing MFCC features.
            use_mfcc: A flag from the model config indicating whether to merge MFCC data.

        Returns:
            A single, merged DataFrame ready for the training pipeline.
        """
        if not use_mfcc or mfcc_df is None or mfcc_df.empty:
            return main_df

        return pd.merge(
            main_df,
            mfcc_df,
            on=Header.UUID.value,
            how='left'
        )
