import math
from typing import List, Dict, Any

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.loudness_analysis_result import \
    LoudnessAnalysisResult


class CrestFactorCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.CREST_FACTOR

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        analysis_result: LoudnessAnalysisResult = data[AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]
        peak = analysis_result.peak
        rms_all = analysis_result.rms_all
        return {AudioDataFeature.CREST_FACTOR: 20 * math.log10(peak / rms_all) if rms_all > 0 else 0.0}
