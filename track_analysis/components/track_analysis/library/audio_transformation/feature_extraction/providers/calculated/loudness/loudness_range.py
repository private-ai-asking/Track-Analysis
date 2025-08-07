from typing import List, Dict, Any

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.intermediary.loudness_analysis_result import \
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
        return {AudioDataFeature.LOUDNESS_RANGE_LU: analysis_result.lra}
