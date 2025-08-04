from typing import List, Dict, Any

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.calculators.helpers.rms_helper import \
    compute_short_time_rms_dbfs
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import AudioDataFeatureCalculator


class MeanRmsCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.AUDIO_SAMPLE_RATE]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.RMS_MEAN

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        rms_metrics = compute_short_time_rms_dbfs(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            samples=data[AudioDataFeature.AUDIO_SAMPLES],
            sr=data[AudioDataFeature.AUDIO_SAMPLE_RATE]
        )
        return {AudioDataFeature.RMS_MEAN: rms_metrics.mean_dbfs}
