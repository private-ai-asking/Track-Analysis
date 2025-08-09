from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.spectral_rolloff import \
    SpectralRolloffExtractor


class SpectralRolloffProvider(AudioDataFeatureProvider):
    """Calculates the mean and standard deviation of the spectral rolloff."""

    def __init__(self, logger, hop_length=512):
        super().__init__()
        self._rolloff_extractor = SpectralRolloffExtractor(logger)
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.HARMONIC_AUDIO]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.SPECTRAL_ROLLOFF_MEAN, AudioDataFeature.SPECTRAL_ROLLOFF_STD]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]
            harmonic = data[AudioDataFeature.HARMONIC_AUDIO]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]

            rolloff_args = {
                "file_path": audio_path,
                "sample_rate": sample_rate,
                "audio": harmonic,
                "start_sample": 0,
                "end_sample": len(harmonic),
                "hop_length": self._hop_length
            }

        spectral_rolloff = self._rolloff_extractor.extract(**rolloff_args, roll_percent=0.85)
        self._add_timed_cache_times(spectral_rolloff)

        with self._measure_processing():
            return {
                AudioDataFeature.SPECTRAL_ROLLOFF_MEAN: float(np.mean(spectral_rolloff.value)),
                AudioDataFeature.SPECTRAL_ROLLOFF_STD: float(np.std(spectral_rolloff.value)),
            }
