from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.intermediary.loudness_analysis_result import \
    LoudnessAnalysisResult


class IntegratedLufsProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.INTEGRATED_LUFS, AudioDataFeature.INTEGRATED_LUFS_STD, AudioDataFeature.INTEGRATED_LUFS_MEAN, AudioDataFeature.INTEGRATED_LUFS_RANGE]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        analysis_result: LoudnessAnalysisResult = data[AudioDataFeature.LOUDNESS_ANALYSIS_RESULT]

        short_term_lufs: np.ndarray = analysis_result.shortterm_lufs

        short_term_lufs_95th = np.percentile(short_term_lufs, 95)
        short_term_lufs_5th = np.percentile(short_term_lufs, 5)

        short_term_lufs_range = short_term_lufs_95th - short_term_lufs_5th

        return {
            AudioDataFeature.INTEGRATED_LUFS: analysis_result.lufs_i,
            AudioDataFeature.INTEGRATED_LUFS_STD: np.std(short_term_lufs),
            AudioDataFeature.INTEGRATED_LUFS_MEAN: np.mean(short_term_lufs),
            AudioDataFeature.INTEGRATED_LUFS_RANGE: short_term_lufs_range,
        }
