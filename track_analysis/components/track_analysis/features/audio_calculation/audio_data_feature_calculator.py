from abc import ABC, abstractmethod
from typing import List, Dict, Any

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature


class AudioDataFeatureCalculator(ABC):
    """A base class for a single-column calculator."""

    @property
    @abstractmethod
    def dependencies(self) -> List[AudioDataFeature]:
        """A list of AudioDataFeature enums that this calculator depends on."""
        pass

    @property
    @abstractmethod
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        """The AudioDataFeature enum for the feature this calculator produces."""
        pass

    @abstractmethod
    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        """
        Performs the calculation for a single track.

        Args:
            data: A dictionary containing the data for the dependencies.

        Returns:
            The calculated value.
        """
        pass
