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

    def __init__(self, logger, number_of_mfccs: int):
        super().__init__()
        self._mfcc_helper = MFCCHelper(MfccExtractor(logger), number_of_mfccs)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MFCC_MEANS, AudioDataFeature.MFCC_STDS, AudioDataFeature.MFCC_VELOCITIES_MEANS, AudioDataFeature.MFCC_VELOCITIES_STDS, AudioDataFeature.MFCC_ACCELERATIONS_MEANS, AudioDataFeature.MFCC_ACCELERATIONS_STDS]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, np.ndarray]:
        results = self._mfcc_helper.get_mffcs(
            audio_path=data[AudioDataFeature.AUDIO_PATH],
            sample_rate=data[AudioDataFeature.SAMPLE_RATE_HZ],
            audio=data[AudioDataFeature.AUDIO_SAMPLES]
        )
        self._add_timed_cache_times(results)
        results = results.value

        with self._measure_processing():
            return {
                AudioDataFeature.MFCC_MEANS: results.means,
                AudioDataFeature.MFCC_STDS: results.stds,
                AudioDataFeature.MFCC_VELOCITIES_MEANS: results.delta_means,
                AudioDataFeature.MFCC_VELOCITIES_STDS: results.delta_stds,
                AudioDataFeature.MFCC_ACCELERATIONS_MEANS: results.delta2_means,
                AudioDataFeature.MFCC_ACCELERATIONS_STDS: results.delta2_stds,
            }
