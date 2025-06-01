import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import VERBOSE

def _generate_subbeat_events(
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

    # include the final beat
    events.append((float(beat_times[-1]), int(beat_frames[-1])))
    return sorted(events, key=lambda evt: evt[0])


class MetricalHierarchyConstructor:
    """
    Constructs a metrical hierarchy by assigning levels to sub-beat events.
    Each event is given a level from 1 to LEVEL_COUNT based on divisors of events per bar.
    """
    LEVEL_COUNT = 4

    def __init__(
            self,
            subdivisions_per_beat: int,
            logger: HoornLogger,
            separator: str,
            cache_dir: Path
    ) -> None:
        self._subdivisions = subdivisions_per_beat
        self._logger = logger
        self._separator = separator

        os.makedirs(cache_dir, exist_ok=True)
        self._compute = Memory(cache_dir, verbose=0).cache(_generate_subbeat_events)

    def construct_hierarchy(
            self,
            beat_times: np.ndarray,
            beat_frames: np.ndarray,
            beats_per_segment: int,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Generate a level array and corresponding event times for sub-beat events.
        """
        self._validate_inputs(beat_times, beat_frames, beats_per_segment)

        events_per_bar = self._calculate_events_per_segment(beats_per_segment)
        top_divisors = self._get_top_divisors(events_per_bar)

        events = self._generate_subbeat_events(beat_times, beat_frames)
        levels, event_times = self._assign_event_levels(
            events, events_per_bar, top_divisors
        )

        self._log_levels(levels)
        return levels, event_times

    def _validate_inputs(
            self,
            beat_times: np.ndarray,
            beat_frames: np.ndarray,
            beats_per_segment: int,
    ) -> None:
        if self._subdivisions < 1:
            raise ValueError("subdivisions_per_beat must be positive")
        if beats_per_segment < 1:
            raise ValueError("beats_per_segment must be positive")
        if beat_times.ndim != 1 or beat_frames.ndim != 1:
            raise ValueError("beat_times and beat_frames must be 1D arrays")
        if beat_times.size != beat_frames.size:
            raise ValueError("beat_times and beat_frames must have the same length")

    def _calculate_events_per_segment(self, beats_per_segment: int) -> int:
        return beats_per_segment * self._subdivisions

    def _get_top_divisors(self, events_per_segment: int) -> List[int]:
        divisors = self._get_divisors(events_per_segment)
        if len(divisors) < self.LEVEL_COUNT:
            raise ValueError(
                f"Insufficient divisors ({len(divisors)}) for {self.LEVEL_COUNT} levels; "
                "increase subdivisions_per_beat or adjust beats_per_bar"
            )
        return divisors[: self.LEVEL_COUNT]

    @staticmethod
    def _get_divisors(n: int) -> List[int]:
        """Return all positive divisors of n, sorted descending."""
        return sorted([d for d in range(1, n + 1) if n % d == 0], reverse=True)

    def _generate_subbeat_events(
            self,
            beat_times: np.ndarray,
            beat_frames: np.ndarray,
    ) -> List[Tuple[float, int]]:
        """
        Linearly interpolate times and frames between beats for subdivisions.
        Returns a sorted list of (time, frame) tuples for each sub-beat event.
        """
        return self._compute(self._subdivisions, beat_times, beat_frames)

    def _assign_event_levels(
            self,
            events: List[Tuple[float, int]],
            events_per_segment: int,
            divisors: List[int],
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Assign levels to each event based on its position modulo events_per_bar.
        Higher levels correspond to coarser metrical positions.
        """
        levels: List[int] = []
        times: List[float] = []

        for idx, (time_point, _) in enumerate(events):
            position = idx % events_per_segment
            level = self._determine_level(position, divisors)
            levels.append(level)
            times.append(time_point)

        return np.array(levels, dtype=int), times

    @staticmethod
    def _determine_level(position: int, divisors: List[int]) -> int:
        """
        Determine event level by finding the first divisor that evenly divides position.
        Level is higher for larger divisors (coarser beats).
        """
        for idx, div in enumerate(divisors):
            if position % div == 0:
                return MetricalHierarchyConstructor.LEVEL_COUNT - idx
        return 1

    def _log_levels(self, levels: np.ndarray) -> None:
        if VERBOSE:
            self._logger.debug(
                f"Assigned levels: {levels.tolist()}", separator=self._separator
            )
