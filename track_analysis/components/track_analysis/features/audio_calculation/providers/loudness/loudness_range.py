from typing import List, Dict, Any

from pyebur128 import get_loudness_range

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.providers.loudness.loudness_analysis_result import \
    LoudnessAnalysisResult


class LoudnessRangeProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.LOUDNESS_RANGE_LU

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        analysis_result: LoudnessAnalysisResult = data[AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]
        return {AudioDataFeature.LOUDNESS_RANGE_LU: get_loudness_range(analysis_result.r128_lra)}
