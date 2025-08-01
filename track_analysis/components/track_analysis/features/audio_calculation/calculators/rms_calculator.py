import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from track_analysis.components.track_analysis.features.audio_calculation.calculators.metric_calculator import \
    AudioMetricCalculator


@dataclass(frozen=True)
class RmsMetrics:
    mean_dbfs: float
    max_dbfs: float
    percentile_90_dbfs: float
    iqr_dbfs: float

def compute_short_time_rms_dbfs(
        samples: np.ndarray,
        sr: int,
        window_ms: float = 50.0,
        hop_ms: float = 10.0
) -> RmsMetrics:
    """
    Compute RMS over short windows and return stats in dBFS.
    """
    # Mono
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    # Convert to samples
    frame_len = int(sr * window_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    if frame_len < 1:
        raise ValueError(f"window_ms {window_ms}ms too small for sr {sr}")
    # Fallback for short audio
    if mono.size < frame_len:
        rms_val = math.sqrt(np.mean(mono**2))
        db = 20 * math.log10(max(rms_val, np.finfo(float).eps))
        return RmsMetrics(db, db, db, 0.0)
    # Framing
    n_frames = 1 + (mono.size - frame_len) // hop_len
    trimmed = mono[: hop_len * n_frames + frame_len - hop_len]
    windows = sliding_window_view(trimmed, frame_len)[::hop_len]
    # RMS per window
    rms_vals = np.sqrt(np.mean(windows**2, axis=1))
    rms_vals = np.clip(rms_vals, np.finfo(rms_vals.dtype).eps, None)
    rms_dbfs = 20 * np.log10(rms_vals)
    # Stats
    p25, p75 = np.percentile(rms_dbfs, [25, 75])
    return RmsMetrics(
        mean_dbfs=float(rms_dbfs.mean()),
        max_dbfs=float(rms_dbfs.max()),
        percentile_90_dbfs=float(np.percentile(rms_dbfs, 90)),
        iqr_dbfs=float(p75 - p25)
    )


class RmsCalculator(AudioMetricCalculator):
    def calculate(self, audio_path: Path, samples: np.ndarray, sr: int, **_):
        rms = compute_short_time_rms_dbfs(samples, sr)
        return {
            "mean_dbfs": rms.mean_dbfs,
            "max_dbfs": rms.max_dbfs,
            "percentile_90_dbfs": rms.percentile_90_dbfs,
            "iqr_dbfs": rms.iqr_dbfs,
        }
