from dataclasses import dataclass

from pyebur128 import R128State


@dataclass(frozen=True)
class LoudnessAnalysisResult:
    """Holds the results of the core loudness analysis pass."""
    r128_i: R128State
    r128_lra: R128State
    r128_tp: R128State
    peak: float
    rms_all: float
    channels: int
