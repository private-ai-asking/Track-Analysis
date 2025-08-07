from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class LoudnessAnalysisResult:
    """Holds the results of the core loudness analysis pass."""
    lufs_i: float
    lra: float
    true_peak: List[float]
    peak: float
    rms_all: float
    channels: int
    shortterm_lufs: np.ndarray
