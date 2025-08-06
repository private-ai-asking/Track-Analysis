from abc import ABC, abstractmethod

import pandas as pd

from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class EnergyCalculator(ABC):
    """
    An interface for using a pre-existing energy model to calculate ratings.
    Implementations are expected to be initialized with a valid EnergyModel.
    """

    @abstractmethod
    def calculate_energy_for_row(self, row: pd.Series) -> float:
        """
        Calculates the energy score for a single track's data.

        Args:
            row: A pandas Series containing the audio features for one track.

        Returns:
            The calculated energy score as a float.
        """
        ...

    @abstractmethod
    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        """
        Calculates energy ratings for a DataFrame using the configured model.

        Args:
            df_to_process: The pandas DataFrame containing the rows to process.
            target_column: The Header enum for the column to add/update with the ratings.

        Returns:
            A new DataFrame with the calculated energy ratings.
        """
        ...
