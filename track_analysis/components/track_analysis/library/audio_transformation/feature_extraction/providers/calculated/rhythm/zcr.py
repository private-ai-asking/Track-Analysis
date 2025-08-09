from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.calculator.zero_crossing import \
    ZeroCrossingRateExtractor


class ZCRProvider(AudioDataFeatureProvider):
    """Calculates the mean Zero Crossing Rate."""

    def __init__(self, logger, hop_length=512):
        super().__init__()
        self._zcr_extractor = ZeroCrossingRateExtractor(logger)
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.ZCR_MEAN, AudioDataFeature.ZCR_STD]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            common_args = {
                "file_path": data[AudioDataFeature.AUDIO_PATH],
                "sample_rate": data[AudioDataFeature.SAMPLE_RATE_HZ],
                "audio": samples,
                "start_sample": 0,
                "end_sample": len(samples),
                "hop_length": self._hop_length
            }

        zcr = self._zcr_extractor.extract(**common_args)
        self._add_timed_cache_times(zcr)

        with self._measure_processing():
            return {
                AudioDataFeature.ZCR_MEAN: float(zcr.value.mean()),
                AudioDataFeature.ZCR_STD: float(zcr.value.std()),
            }
