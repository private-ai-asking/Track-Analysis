from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.calculators.helpers.mfcc_helper import \
    MFCCHelper
from track_analysis.components.track_analysis.features.core.cacheing.mfcc import MfccExtractor


class MfccCalculator(AudioDataFeatureCalculator):
    """Calculates the MFCC means and standard deviations."""

    def __init__(self, logger):
        # The helper now takes the extractor directly
        self._mfcc_helper = MFCCHelper(MfccExtractor(logger))

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLE_RATE, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MFCC_MEANS, AudioDataFeature.MFCC_STDS]

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[str, np.ndarray]:
        mfcc_means, mfcc_stds = self._mfcc_helper.get_mffcs(
            audio_path=data[AudioDataFeature.AUDIO_PATH],
            sample_rate=data[AudioDataFeature.AUDIO_SAMPLE_RATE],
            audio=data[AudioDataFeature.AUDIO_SAMPLES]
        )
        return {
            AudioDataFeature.MFCC_MEANS.name: mfcc_means,
            AudioDataFeature.MFCC_STDS.name: mfcc_stds,
        }
