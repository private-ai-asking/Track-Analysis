from pathlib import Path
from typing import List, Tuple

import numpy as np

from track_analysis.components.track_analysis.features.core.caching.cached_operations.shared import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["beat_times", "beat_frames"])
def generate_subbeat_events(
        file_path: Path,
        subdivisions: int,
        beat_times: np.ndarray,
        beat_frames: np.ndarray) -> List[Tuple[float, int]]:
    events: List[Tuple[float, int]] = []

    for i in range(beat_times.size - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        f0, f1 = beat_frames[i], beat_frames[i + 1]
        interval = t1 - t0
        frame_span = f1 - f0

        for sub in range(subdivisions):
            frac = sub / subdivisions
            time_point = t0 + frac * interval
            frame_point = int(f0 + frac * frame_span)
            events.append((time_point, frame_point))  # type: ignore

    events.append((float(beat_times[-1]), int(beat_frames[-1])))
    return sorted(events, key=lambda evt: evt[0])
