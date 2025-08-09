import dataclasses
from typing import Any, Dict

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    ProviderProcessingStatistics, AudioDataFeatureProvider


@dataclasses.dataclass(frozen=True)
class TrackProcessingResult:
    retrieved_features: Dict[AudioDataFeature, Any]
    provider_stats: Dict[AudioDataFeatureProvider, ProviderProcessingStatistics]
