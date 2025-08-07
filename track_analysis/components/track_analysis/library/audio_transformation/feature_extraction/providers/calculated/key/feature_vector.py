from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.feature.lof.lof_feature_transformer import \
    LOFFeatureTransformer
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.feature.vector.feature_vector_extractor import \
    FeatureVectorExtractor
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.profiling.segment_profiler import \
    ProfiledSegment


class FeatureVectorProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger):
        self._feature_extractor: FeatureVectorExtractor = FeatureVectorExtractor(
            logger,
            transformer=LOFFeatureTransformer()
        )

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_SEGMENTS_PROFILED]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_FEATURE_VECTOR]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        segments: List[ProfiledSegment] = data[AudioDataFeature.TRACK_SEGMENTS_PROFILED]
        track_feature_vector = self._feature_extractor.extract_segments(segments)

        return {
            AudioDataFeature.TRACK_FEATURE_VECTOR: track_feature_vector,
        }
