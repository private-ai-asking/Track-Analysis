from abc import ABC, abstractmethod
from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature


class AudioDataFeatureProvider(ABC):
    """A base class for a single-feature provider."""

    @property
    @abstractmethod
    def dependencies(self) -> List[AudioDataFeature]:
        """A list of AudioDataFeature enums that this provider depends on."""
        pass

    @property
    @abstractmethod
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        """The AudioDataFeature enum for the feature this provider produces."""
        pass

    @abstractmethod
    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        """
        Performs the retrieval operation for a single track.

        Args:
            data: A dictionary containing the data for the dependencies.

        Returns:
            The retrieved value.
        """
        pass
