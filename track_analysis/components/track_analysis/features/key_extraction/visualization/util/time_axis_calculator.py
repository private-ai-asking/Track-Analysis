import numpy as np


class TimeAxisCalculator:
    """Computes a time axis (in seconds) for frame-based data."""
    def __init__(self, hop_length: int, sample_rate: int) -> None:
        self._hop_length = hop_length
        self._sample_rate = sample_rate

    def compute(self, frames: int) -> np.ndarray:
        return np.arange(frames) * (self._hop_length / self._sample_rate)
