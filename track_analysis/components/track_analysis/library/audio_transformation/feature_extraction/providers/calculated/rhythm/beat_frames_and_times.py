from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.tempo_provider import \
    BeatDetector


class BeatFramesAndTimesProvider(AudioDataFeatureProvider):
    """
    Provides the frame indices of detected beats using the BeatDetector.
    """
    def __init__(self, beat_detector: BeatDetector, hop_length: int = 512):
        self._beat_detector = beat_detector
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.SAMPLE_RATE_HZ,
            AudioDataFeature.ONSET_ENVELOPE,
            AudioDataFeature.BPM,
        ]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.BEAT_FRAMES, AudioDataFeature.BEAT_TIMES]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        frames, times = self._beat_detector.get_beat_frames_and_times(
            audio_path=data[AudioDataFeature.AUDIO_PATH],
            audio=data[AudioDataFeature.AUDIO_SAMPLES],
            sample_rate=data[AudioDataFeature.SAMPLE_RATE_HZ],
            onset_envelope=data[AudioDataFeature.ONSET_ENVELOPE],
            hop_length=self._hop_length,
            tempo=data.get(AudioDataFeature.BPM)
        )
        return {
            AudioDataFeature.BEAT_FRAMES: frames
            ,AudioDataFeature.BEAT_TIMES: times
        }
