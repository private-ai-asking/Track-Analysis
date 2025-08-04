from typing import List, Dict, Any

from pyebur128 import get_loudness_global

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.loudness_analysis_result import \
    LoudnessAnalysisResult


class IntegratedLufsCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.INTEGRATED_LUFS

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        analysis_result: LoudnessAnalysisResult = data[AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]
        return {AudioDataFeature.INTEGRATED_LUFS: get_loudness_global(analysis_result.r128_i)}
