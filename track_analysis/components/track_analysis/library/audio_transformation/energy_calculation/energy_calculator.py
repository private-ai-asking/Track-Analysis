from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature


class EnergyAlgorithm(ABC):
    """
    An interface for using a pre-existing energy model to calculate ratings.
    Implementations are expected to be initialized with a valid EnergyModel.
    """

    # TODO - Resolve this. The energy calculator should not be aware of audio data features.
    @abstractmethod
    def get_dependencies(self) -> List[AudioDataFeature]:
        """
        Exposes the model's required features as AudioDataFeature enums.
        This is the single source of truth for the provider's dependencies.
        """

    @abstractmethod
    def calculate_energy_for_row(self, row_df: pd.DataFrame) -> float:
        """
        Calculates the energy score for a single track's data.

        Args:
            row_df: The row df.

        Returns:
            The calculated energy score as a float.
        """
        ...
