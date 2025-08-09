from abc import ABC, abstractmethod
from typing import Dict, List, Any

from track_analysis.components.track_analysis.features.data_generation.processors.main_feature_processor import \
    TrackProcessingResult
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature


class OrchestratorInterface(ABC):
    """
    Abstract base class for all feature provider orchestrators.

    This defines a common `async` interface, allowing different execution
    strategies (e.g., synchronous, asynchronous) to be used interchangeably.
    """
    @abstractmethod
    async def process_track(
            self,
            initial_data: Dict[AudioDataFeature, Any],
            features_to_calculate: List[AudioDataFeature]
    ) -> TrackProcessingResult:
        """
        Asynchronously processes a single track to calculate the requested features.

        Args:
            initial_data: A dictionary of base features required to start processing.
            features_to_calculate: A list of the final features the client wants.

        Returns:
            A TrackProcessingResult containing the final features and timing stats.
        """
        pass
