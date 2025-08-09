from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.helpers.rms_helper import \
    compute_short_time_rms_dbfs


class RMSProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.SAMPLE_RATE_HZ]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.RMS_MEAN, AudioDataFeature.RMS_MAX, AudioDataFeature.RMS_IQR, AudioDataFeature.RMS_PERC]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]

        rms_metrics_result = compute_short_time_rms_dbfs(
            file_path=audio_path,
            samples=samples,
            sr=sample_rate
        )
        self._add_timed_cache_times(rms_metrics_result)

        return {
            AudioDataFeature.RMS_MEAN: rms_metrics_result.value.mean_dbfs,
            AudioDataFeature.RMS_MAX: rms_metrics_result.value.max_dbfs,
            AudioDataFeature.RMS_IQR: rms_metrics_result.value.iqr_dbfs,
            AudioDataFeature.RMS_PERC: rms_metrics_result.value.percentile_90_dbfs
        }
