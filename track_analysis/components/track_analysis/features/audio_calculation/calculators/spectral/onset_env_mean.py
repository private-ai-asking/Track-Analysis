from typing import List, Dict, Any

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature


class OnsetEnvMeanCalculator(AudioDataFeatureCalculator):
    """Calculates the mean of the global onset strength envelope."""

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.ONSET_ENVELOPE]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_GLOBAL

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        env_global = data[AudioDataFeature.ONSET_ENVELOPE]
        return {AudioDataFeature.ONSET_ENV_MEAN_GLOBAL: float(env_global.mean())}
