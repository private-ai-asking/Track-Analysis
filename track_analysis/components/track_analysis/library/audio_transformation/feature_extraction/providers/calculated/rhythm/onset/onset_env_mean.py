from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature


class OnsetEnvMeanProvider(AudioDataFeatureProvider):
    """Calculates the mean of the global onset strength envelope."""

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.ONSET_ENVELOPE]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_GLOBAL

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            env_global = data[AudioDataFeature.ONSET_ENVELOPE]
            return {AudioDataFeature.ONSET_ENV_MEAN_GLOBAL: float(env_global.mean())}
