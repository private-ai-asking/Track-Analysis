from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.harmonicity import \
    HarmonicityExtractor


class HarmonicityProvider(AudioDataFeatureProvider):
    """Calculates the harmonicity of a track."""

    def __init__(self, logger):
        super().__init__()
        self._harmonicity_extractor = HarmonicityExtractor(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.SAMPLE_RATE_HZ,
            AudioDataFeature.HARMONIC_AUDIO,
            AudioDataFeature.AUDIO_SAMPLES,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.HARMONICITY

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
            harmonic = data[AudioDataFeature.HARMONIC_AUDIO]
            audio_path = data[AudioDataFeature.AUDIO_PATH]

        harmonicity = self._harmonicity_extractor.extract(
            file_path=audio_path,
            sample_rate=sample_rate,
            start_sample=0,
            end_sample=len(samples),
            harmonic=harmonic,
            full_audio=samples,
        )
        self._add_timed_cache_times(harmonicity)

        with self._measure_processing():
            return {
                AudioDataFeature.HARMONICITY: harmonicity.value,
            }
