from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.calculator.onset_envelope import \
    OnsetStrengthExtractor


class OnsetEnvelopeProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, hop_length: int = 512):
        super().__init__()
        self._hop_length = hop_length
        self._onset_extractor: OnsetStrengthExtractor = OnsetStrengthExtractor(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.AUDIO_SAMPLES]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.ONSET_ENVELOPE

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
            samples = data[AudioDataFeature.AUDIO_SAMPLES]

        envelope_results = self._onset_extractor.extract_envelope(
            file_path=audio_path,
            sample_rate=sample_rate,
            start_sample=0,
            end_sample=len(samples),
            hop_length=self._hop_length,
            audio=samples,
            unique_string="onset-envelope-full-audio"
        )
        self._add_timed_cache_times(envelope_results)

        with self._measure_processing():
            return {
                AudioDataFeature.ONSET_ENVELOPE: envelope_results.value,
            }

class PercussiveOnsetEnvelopeProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, hop_length: int = 512):
        super().__init__()
        self._hop_length = hop_length
        self._onset_extractor: OnsetStrengthExtractor = OnsetStrengthExtractor(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.PERCUSSIVE_AUDIO]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.PERCUSSIVE_ONSET_ENVELOPE

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
            samples = data[AudioDataFeature.PERCUSSIVE_AUDIO]

        envelope_results = self._onset_extractor.extract_envelope(
            file_path=audio_path,
            sample_rate=sample_rate,
            start_sample=0,
            end_sample=len(samples),
            hop_length=self._hop_length,
            audio=samples,
            unique_string="onset-envelope-percussive-v2"
        )
        self._add_timed_cache_times(envelope_results)

        with self._measure_processing():
            return {
                AudioDataFeature.PERCUSSIVE_ONSET_ENVELOPE: envelope_results.value,
            }
