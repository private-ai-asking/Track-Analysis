import math
from typing import List, Dict, Any

from pyebur128 import get_true_peak

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.loudness_analysis_result import \
    LoudnessAnalysisResult


class TruePeakCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.TRUE_PEAK

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        analysis_result: LoudnessAnalysisResult = data[AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]
        tp_ch = [
            20 * math.log10(get_true_peak(analysis_result.r128_tp, ch))
            for ch in range(analysis_result.channels)
        ]
        return {AudioDataFeature.TRUE_PEAK: max(tp_ch) if tp_ch else 0.0}
