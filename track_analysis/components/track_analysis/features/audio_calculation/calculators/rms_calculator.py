from dataclasses import dataclass
from pathlib import Path

import numpy as np

from track_analysis.components.track_analysis.features.audio_calculation.calculators.metric_calculator import \
    AudioMetricCalculator
from track_analysis.components.track_analysis.features.core.cacheing.rms import compute_linear_rms, \
    compute_linear_rms_cached


@dataclass(frozen=True)
class RmsMetrics:
    mean_dbfs: float
    max_dbfs: float
    percentile_90_dbfs: float
    iqr_dbfs: float

def compute_short_time_rms_dbfs(
        file_path: Path,
        samples: np.ndarray,
        sr: int,
        window_ms: float = 50.0,
        hop_ms: float = 10.0
) -> RmsMetrics:
    """
    Compute RMS over short windows and return stats in dBFS.
    """
    # Use the reusable helper to get linear RMS values
    rms_vals = compute_linear_rms_cached(file_path=file_path, start_sample=0, end_sample=len(samples), sample_rate=sr, window_ms=window_ms, hop_ms=hop_ms, audio=samples, method_string="full-audio")

    # Convert to dBFS
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
        rms = compute_short_time_rms_dbfs(file_path=audio_path, samples=samples, sr=sr)
        return {
            "mean_dbfs": rms.mean_dbfs,
            "max_dbfs": rms.max_dbfs,
            "percentile_90_dbfs": rms.percentile_90_dbfs,
            "iqr_dbfs": rms.iqr_dbfs,
        }
