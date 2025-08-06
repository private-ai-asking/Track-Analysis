from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature


class TempoVariationProvider(AudioDataFeatureProvider):
    """Calculates the standard deviation of the dynamic tempo."""

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.DYNAMIC_TEMPO]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.TEMPO_VARIATION

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        dynamic_tempo = data[AudioDataFeature.DYNAMIC_TEMPO]
        return {AudioDataFeature.TEMPO_VARIATION: np.std(dynamic_tempo)}
