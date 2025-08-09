from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature


class OnsetRateProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        # Depends on the pre-calculated onset peaks
        return [AudioDataFeature.ONSET_PEAKS, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.SAMPLE_RATE_HZ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_GLOBAL

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            peaks_global = data[AudioDataFeature.ONSET_PEAKS]
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sr = data[AudioDataFeature.SAMPLE_RATE_HZ]

            duration_sec = len(samples) / sr if sr > 0 else 1.0
            return {AudioDataFeature.ONSET_RATE_GLOBAL: len(peaks_global) / duration_sec}
