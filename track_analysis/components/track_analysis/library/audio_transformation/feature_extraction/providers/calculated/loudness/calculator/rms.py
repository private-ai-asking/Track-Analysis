import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


def compute_linear_rms(
        samples: np.ndarray,
        sr: int,
        window_ms: float = 50.0,
        hop_ms: float = 10.0
) -> np.ndarray:
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
        return np.array([rms_val])
    # Framing
    n_frames = 1 + (mono.size - frame_len) // hop_len
    trimmed = mono[: hop_len * n_frames + frame_len - hop_len]
    windows = sliding_window_view(trimmed, frame_len)[::hop_len]
    # RMS per window
    rms_vals = np.sqrt(np.mean(windows**2, axis=1))
    return rms_vals

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["audio"])
def compute_linear_rms_cached(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        method_string: str,
        window_ms:     float = 50.0,
        hop_ms:        float = 10.0,
        audio:         np.ndarray = None,
) -> TimedCacheResult[np.ndarray]:
    audio = audio[start_sample:end_sample]

    # noinspection PyTypeChecker
    return compute_linear_rms(
        samples=audio,
        sr=sample_rate,
        window_ms=window_ms,
        hop_ms=hop_ms,
    )
