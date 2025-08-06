from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.mfcc import \
    MfccExtractor
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.helpers.mfcc_helper import \
    MFCCHelper


class MfccProvider(AudioDataFeatureProvider):
    """Calculates the MFCC means and standard deviations."""

    def __init__(self, logger):
        self._mfcc_helper = MFCCHelper(MfccExtractor(logger))

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MFCC_MEANS, AudioDataFeature.MFCC_STDS]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, np.ndarray]:
        mfcc_means, mfcc_stds = self._mfcc_helper.get_mffcs(
            audio_path=data[AudioDataFeature.AUDIO_PATH],
            sample_rate=data[AudioDataFeature.SAMPLE_RATE_HZ],
            audio=data[AudioDataFeature.AUDIO_SAMPLES]
        )
        return {
            AudioDataFeature.MFCC_MEANS: mfcc_means,
            AudioDataFeature.MFCC_STDS: mfcc_stds,
        }
