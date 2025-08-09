from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.segmentation.audio_segmenter import \
    AudioSegmenter


class TrackRawSegmentProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, subdivisions_per_beat: int = 2, hop_length_samples: int = 512,
                 beats_per_segment: int = 8, min_segment_beat_level: int = 3):
        super().__init__()
        self._min_segment_beat_level = min_segment_beat_level

        self._audio_segmenter: AudioSegmenter = AudioSegmenter(
            logger=logger,
            subdivisions_per_beat=subdivisions_per_beat,
            hop_length_samples=hop_length_samples,
            beats_per_segment=beats_per_segment
        )

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.SAMPLE_RATE_HZ,
            AudioDataFeature.SUB_BEAT_EVENTS,
        ]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_SEGMENTS_RAW]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
            sub_beat_events = data[AudioDataFeature.SUB_BEAT_EVENTS]

            audio_segments = self._audio_segmenter.get_segments(
                audio_samples=audio_samples,
                sample_rate=sample_rate,
                min_segment_level=self._min_segment_beat_level,
                sub_beat_events=sub_beat_events
            )

            return {
                AudioDataFeature.TRACK_SEGMENTS_RAW: audio_segments,
            }
