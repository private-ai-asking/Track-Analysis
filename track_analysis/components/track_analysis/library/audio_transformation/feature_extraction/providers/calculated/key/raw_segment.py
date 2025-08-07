from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.segmentation.audio_segmenter import \
    AudioSegmenter


class TrackRawSegmentProvider(AudioDataFeatureProvider):
    def __init__(self,
                 logger: HoornLogger,
                 hop_length_samples: int = 512,
                 subdivisions_per_beat: int = 2,
                 beats_per_segment: int = 8,
                 min_segment_beat_level: int = 3):
        self._min_segment_beat_level = min_segment_beat_level

        self._audio_segmenter: AudioSegmenter = AudioSegmenter(
            logger, hop_length_samples, subdivisions_per_beat, beats_per_segment
        )

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.BEAT_FRAMES,
            AudioDataFeature.BEAT_TIMES,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.SAMPLE_RATE_HZ
        ]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_SEGMENTS_RAW]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        beat_frames = data[AudioDataFeature.BEAT_FRAMES]
        beat_times = data[AudioDataFeature.BEAT_TIMES]
        audio_samples = data[AudioDataFeature.AUDIO_SAMPLES]
        sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]

        audio_segments = self._audio_segmenter.get_segments(
            audio_path=audio_path,
            beat_frames=beat_frames,
            beat_times=beat_times,
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            min_segment_level=self._min_segment_beat_level
        )

        return {
            AudioDataFeature.TRACK_SEGMENTS_RAW: audio_segments,
        }
