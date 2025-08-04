from typing import List, Dict, Any

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.core.cacheing.harmonicity import HarmonicityExtractor


class HarmonicityProvider(AudioDataFeatureProvider):
    """Calculates the harmonicity of a track."""

    def __init__(self, logger):
        self._harmonicity_extractor = HarmonicityExtractor(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.AUDIO_SAMPLE_RATE,
            AudioDataFeature.HARMONIC_AUDIO,
            AudioDataFeature.AUDIO_SAMPLES,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.HARMONICITY

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        harmonicity = self._harmonicity_extractor.extract(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            sample_rate=data[AudioDataFeature.AUDIO_SAMPLE_RATE],
            start_sample=0,
            end_sample=len(samples),
            harmonic=data[AudioDataFeature.HARMONIC_AUDIO],
            full_audio=samples,
        )

        return {
            AudioDataFeature.HARMONICITY: harmonicity,
        }
