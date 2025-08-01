import math
from pathlib import Path

import numpy as np
from pyebur128 import get_loudness_global, get_loudness_range, get_true_peak, R128State, MeasurementMode

from track_analysis.components.track_analysis.features.audio_calculation.calculators.metric_calculator import \
    AudioMetricCalculator


class LoudnessCalculator(AudioMetricCalculator):
    def calculate(self, audio_path: Path, samples: np.ndarray, sr: int, chunk_size=4096, **_):
        # Ensure two-dimensional
        if samples.ndim == 1:
            samples = samples[:, None]
        frames, channels = samples.shape
        st_i   = R128State(channels, sr, MeasurementMode.MODE_I)
        st_lra = R128State(channels, sr, MeasurementMode.MODE_LRA)
        st_tp  = R128State(channels, sr, MeasurementMode.MODE_TRUE_PEAK)
        sum_sq = 0.0
        max_sq = 0.0
        for off in range(0, frames, chunk_size):
            block = samples[off:off+chunk_size]
            flat  = block.flatten()
            n = block.shape[0]
            st_i.add_frames(flat, n)
            st_lra.add_frames(flat, n)
            st_tp.add_frames(flat, n)
            sq = block**2
            sum_sq += sq.sum()
            max_sq  = max(max_sq, sq.max())
        lufs = get_loudness_global(st_i)
        lra  = get_loudness_range(st_lra)
        tp_ch = [20*math.log10(get_true_peak(st_tp, ch)) for ch in range(channels)]
        tp    = max(tp_ch) if tp_ch else 0.0
        peak  = math.sqrt(max_sq)
        rms_all = math.sqrt(sum_sq/(frames*channels)) if frames>0 else 0.0
        crest = 20*math.log10(peak/rms_all) if rms_all>0 else 0.0
        return {
            "true_peak_dbtp": tp,
            "integrated_lufs": lufs,
            "loudness_range_lu": lra,
            "crest_factor_db": crest
        }
