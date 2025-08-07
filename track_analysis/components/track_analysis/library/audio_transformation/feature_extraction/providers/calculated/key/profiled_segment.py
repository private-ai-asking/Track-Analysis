from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.profiling.segment_profiler import \
    SegmentProfiler


class TrackProfiledSegmentProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger):
        self._segment_profiler: SegmentProfiler = SegmentProfiler(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_NOTE_EVENTS, AudioDataFeature.TRACK_SEGMENTS_RAW]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_SEGMENTS_PROFILED]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        raw_segments = data[AudioDataFeature.TRACK_SEGMENTS_RAW]
        note_events = data[AudioDataFeature.TRACK_NOTE_EVENTS]

        profiled_segments = self._segment_profiler.profile_segments(
            raw_segments,
            note_events,
        )

        return {
            AudioDataFeature.TRACK_SEGMENTS_PROFILED: profiled_segments,
        }
