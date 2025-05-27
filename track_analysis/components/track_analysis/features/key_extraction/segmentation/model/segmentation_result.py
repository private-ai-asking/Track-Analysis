from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    """
    Container for segmented audio data and metadata.
    """
    segments: List[np.ndarray]
    start_times: List[float]
    durations: List[float]
