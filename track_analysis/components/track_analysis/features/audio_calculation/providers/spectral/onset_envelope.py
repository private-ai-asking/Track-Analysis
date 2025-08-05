from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.core.caching.cached_operations.onset_envelope import OnsetStrengthExtractor


class OnsetEnvelopeProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, hop_length: int = 512):
        self._hop_length = hop_length
        self._onset_extractor: OnsetStrengthExtractor = OnsetStrengthExtractor(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.ONSET_ENVELOPE

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        envelope = self._onset_extractor.extract(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            sample_rate=data[AudioDataFeature.SAMPLE_RATE_HZ],
            start_sample=0,
            end_sample=len(data[AudioDataFeature.AUDIO_SAMPLES]),
            hop_length=self._hop_length,
            audio=data[AudioDataFeature.AUDIO_SAMPLES]
        )

        return {
            AudioDataFeature.ONSET_ENVELOPE: envelope,
        }
