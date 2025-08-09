from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.calculator.onset_envelope import \
    OnsetStrengthExtractor


class DynamicTempoProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, hop_length: int = 512):
        super().__init__()
        self._hop_length = hop_length
        self._onset_extractor: OnsetStrengthExtractor = OnsetStrengthExtractor(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.DYNAMIC_TEMPO

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        dynamic_tempo = self._onset_extractor.extract_dynamic_tempo(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            sample_rate=data[AudioDataFeature.SAMPLE_RATE_HZ],
            start_sample=0,
            end_sample=len(data[AudioDataFeature.AUDIO_SAMPLES]),
            hop_length=self._hop_length,
            audio=data[AudioDataFeature.AUDIO_SAMPLES]
        )
        self._add_timed_cache_times(dynamic_tempo)

        return {
            AudioDataFeature.DYNAMIC_TEMPO: dynamic_tempo.value,
        }
