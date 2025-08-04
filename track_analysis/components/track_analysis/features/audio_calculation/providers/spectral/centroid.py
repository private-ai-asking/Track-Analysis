from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature


class SpectralCentroidProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.SPECTRAL_CENTROID_ARRAY]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.SPECTRAL_CENTROID_MEAN, AudioDataFeature.SPECTRAL_CENTROID_MAX, AudioDataFeature.SPECTRAL_CENTROID_STD]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, float]:
        array = data[AudioDataFeature.SPECTRAL_CENTROID_ARRAY]
        mean = np.mean(array)
        max = np.max(array)
        std = np.std(array)

        return {
            AudioDataFeature.SPECTRAL_CENTROID_MEAN: float(mean),
            AudioDataFeature.SPECTRAL_CENTROID_MAX: float(max),
            AudioDataFeature.SPECTRAL_CENTROID_STD: float(std),
        }
