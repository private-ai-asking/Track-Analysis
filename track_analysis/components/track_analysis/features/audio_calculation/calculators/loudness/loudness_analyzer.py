import math
from typing import List, Dict, Any

import numpy as np
from pyebur128 import R128State, MeasurementMode

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.loudness_analysis_result import \
    LoudnessAnalysisResult


class LoudnessAnalyzer(AudioDataFeatureCalculator):
    """
    Performs the core R128 analysis pass over the audio samples.
    This is an intermediate calculator that other loudness calculators depend on.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.AUDIO_SAMPLE_RATE]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.LOUDNESS_ANALYSIS_RESULT

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        sr = data[AudioDataFeature.AUDIO_SAMPLE_RATE]
        chunk_size = 4096

        if samples.ndim == 1:
            samples = samples[:, np.newaxis]

        frames, channels = samples.shape
        st_i = R128State(channels, sr, MeasurementMode.MODE_I)
        st_lra = R128State(channels, sr, MeasurementMode.MODE_LRA)
        st_tp = R128State(channels, sr, MeasurementMode.MODE_TRUE_PEAK)

        sum_sq = 0.0
        max_sq = 0.0

        for off in range(0, frames, chunk_size):
            block = samples[off : off + chunk_size]
            flat = block.flatten()
            n = block.shape[0]
            st_i.add_frames(flat, n)
            st_lra.add_frames(flat, n)
            st_tp.add_frames(flat, n)
            sq = block**2
            sum_sq += sq.sum()
            max_sq = max(max_sq, sq.max())

        peak = math.sqrt(max_sq)
        rms_all = math.sqrt(sum_sq / (frames * channels)) if frames > 0 else 0.0

        return {AudioDataFeature.LOUDNESS_ANALYSIS_RESULT: LoudnessAnalysisResult(
            r128_i=st_i,
            r128_lra=st_lra,
            r128_tp=st_tp,
            peak=peak,
            rms_all=rms_all,
            channels=channels,
        )}
