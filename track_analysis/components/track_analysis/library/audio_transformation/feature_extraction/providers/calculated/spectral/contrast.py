from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.spectral_contrast import \
    SpectralContrastExtractor


class SpectralContrastProvider(AudioDataFeatureProvider):
    """Calculates the mean spectral contrast from the full audio signal."""

    def __init__(self, logger, hop_length=512):
        self._contrast_extractor = SpectralContrastExtractor(logger)
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.SPECTRAL_CONTRAST_MEAN, AudioDataFeature.SPECTRAL_CONTRAST_STD]

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
            AudioDataFeature.SPECTRAL_CONTRAST_MEAN: float(contrast.mean()),
            AudioDataFeature.SPECTRAL_CONTRAST_STD: float(contrast.std()),
        }
