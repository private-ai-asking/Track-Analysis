from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ProcessedFeature:
    name: str = ""
    total_time: float = 0.0
    wait_time: float = 0.0
    process_time: float = 0.0
    call_count: int = 0
    avg_time_ms: float = 0.0
    all_timings_ms: List[float] = field(default_factory=list)
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    stdev_ms: float = 0.0

    def finalize_stats(self):
        """Calculates final stats after all data is aggregated."""
        if not self.all_timings_ms:
            return

        self.total_time = sum(self.all_timings_ms) / 1000.0

        self.avg_time_ms = (self.total_time / self.call_count * 1000) if self.call_count > 0 else 0

        self.min_time_ms = min(self.all_timings_ms)
        self.max_time_ms = max(self.all_timings_ms)
        if len(self.all_timings_ms) > 1:
            self.stdev_ms = np.std(self.all_timings_ms) # type: ignore
