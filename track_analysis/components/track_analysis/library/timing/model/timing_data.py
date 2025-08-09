import dataclasses
from typing import List

from track_analysis.components.track_analysis.library.timing.model.feature_timing_data import FeatureTimingData


@dataclasses.dataclass(frozen=True)
class TimingData:
    total_time_spent: float
    """The total time the execution took (including waiting, processing, AND sub-features)."""

    own_waiting_time: float
    """The total time THIS function spent waiting."""

    own_processing_time: float
    """The total time THIS function spent processing."""

    feature_dissection: List[FeatureTimingData]
    """A list of more comprehensive feature execution data."""
