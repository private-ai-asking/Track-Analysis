from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.helpers.rms_helper import \
    compute_short_time_rms_dbfs


class PercentileRmsProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.SAMPLE_RATE_HZ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.RMS_PERC

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        rms_metrics = compute_short_time_rms_dbfs(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            samples=data[AudioDataFeature.AUDIO_SAMPLES],
            sr=data[AudioDataFeature.SAMPLE_RATE_HZ]
        )
        return {AudioDataFeature.RMS_PERC: rms_metrics.percentile_90_dbfs}
