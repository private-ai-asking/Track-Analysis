from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature


class SpectralFluxCalculator(AudioDataFeatureCalculator):
    """Calculates the mean and max of the spectral flux."""

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.SPECTRAL_FLUX_ARRAY]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.SPECTRAL_FLUX_MEAN, AudioDataFeature.SPECTRAL_FLUX_MAX]

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, float]:
        return {
            AudioDataFeature.SPECTRAL_FLUX_MEAN: float(np.mean(data[AudioDataFeature.SPECTRAL_FLUX_ARRAY])),
            AudioDataFeature.SPECTRAL_FLUX_MAX: float(np.max(data[AudioDataFeature.SPECTRAL_FLUX_ARRAY])),
            AudioDataFeature.SPECTRAL_FLUX_STD: float(np.std(data[AudioDataFeature.SPECTRAL_FLUX_ARRAY])),
        }
