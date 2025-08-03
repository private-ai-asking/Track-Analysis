from abc import ABC, abstractmethod

import pandas as pd

from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class EnergyCalculator(ABC):
    @abstractmethod
    def calculate_energy_for_row(self, row: pd.Series) -> float:
        """Calculates the energy based on the audio features."""

    @abstractmethod
    def calculate_ratings_for_df(self, df_to_process: pd.DataFrame, target_column: Header) -> pd.DataFrame:
        """
        Calculates energy ratings for a given DataFrame and returns it with a new/updated column.

        This method is stateless and does not modify the class's internal data. It is the
        recommended method for batch processing.

        Args:
            df_to_process: The pandas DataFrame containing the rows to process.
            target_column: The Header enum for the column to add/update with the ratings.

        Returns:
            A new DataFrame with the calculated energy ratings.
        """
