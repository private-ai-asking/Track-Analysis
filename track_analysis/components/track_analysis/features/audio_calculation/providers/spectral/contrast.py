from typing import List, Dict, Any

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.core.caching.cached_operations.spectral_contrast import SpectralContrastExtractor


class SpectralContrastProvider(AudioDataFeatureProvider):
    """Calculates the mean spectral contrast from the full audio signal."""

    def __init__(self, logger, hop_length=512):
        self._contrast_extractor = SpectralContrastExtractor(logger)
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.SPECTRAL_CONTRAST_MEAN

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        common_args = {
            "file_path": data[AudioDataFeature.AUDIO_PATH],
            "sample_rate": data[AudioDataFeature.SAMPLE_RATE_HZ],
            "audio": samples,
            "start_sample": 0,
            "end_sample": len(samples),
            "hop_length": self._hop_length
        }
        contrast = self._contrast_extractor.extract(**common_args)
        return {
            AudioDataFeature.SPECTRAL_CONTRAST_MEAN: float(contrast.mean())
        }
