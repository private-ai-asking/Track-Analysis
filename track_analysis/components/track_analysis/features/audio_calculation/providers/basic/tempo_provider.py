from typing import Dict, Any, List

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.core.caching.cached_operations.beat import BeatDetector


class TempoProvider(AudioDataFeatureProvider):
    def __init__(self, beat_detector: BeatDetector, hop_length: int):
        self._beat_detector = beat_detector
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.ONSET_ENVELOPE]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.BPM

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
        onset_envelope = data[AudioDataFeature.ONSET_ENVELOPE]

        tempo = self._beat_detector.get_tempo(audio_path=audio_path, audio=samples, sample_rate=sample_rate, onset_envelope=onset_envelope, hop_length=self._hop_length)

        return {
            AudioDataFeature.BPM: tempo,
        }
