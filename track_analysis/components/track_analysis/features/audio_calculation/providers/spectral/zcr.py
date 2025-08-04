from typing import List, Dict, Any

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.core.cacheing.zero_crossing import ZeroCrossingRateExtractor


class ZCRProvider(AudioDataFeatureProvider):
    """Calculates the mean Zero Crossing Rate."""

    def __init__(self, logger, hop_length=512):
        self._zcr_extractor = ZeroCrossingRateExtractor(logger)
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLE_RATE, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ZCR_MEAN

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        common_args = {
            "file_path": data[AudioDataFeature.AUDIO_PATH],
            "sample_rate": data[AudioDataFeature.AUDIO_SAMPLE_RATE],
            "audio": samples,
            "start_sample": 0,
            "end_sample": len(samples),
            "hop_length": self._hop_length
        }
        zcr = self._zcr_extractor.extract(**common_args)
        return {AudioDataFeature.ZCR_MEAN: float(zcr.mean())}
