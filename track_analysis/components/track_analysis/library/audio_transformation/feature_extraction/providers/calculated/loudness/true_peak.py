import math
from typing import List, Dict, Any

from pyebur128 import get_true_peak

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.intermediary.loudness_analysis_result import \
    LoudnessAnalysisResult


class TruePeakProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.TRUE_PEAK

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            analysis_result: LoudnessAnalysisResult = data[AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]
            tp_ch = [
                20 * math.log10(tp)
                for tp in analysis_result.true_peak
            ]
            return {AudioDataFeature.TRUE_PEAK: max(tp_ch) if tp_ch else 0.0}
